import argparse
from trt_utils.segcolors import lanecolor, midcolor, colors, obstacle
from torchvision import transforms
from PIL import Image
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import pdb
import os
import cv2
import time

class TRTSegmentor(object):
    def __init__(self, 
        onnxpath, 
        colors,
        lanecolor,
        midcolor,
        insize=(360,360),
        outsize=(360,360),
        maxworkspace=(1<<30), 
        precision='FP16', 
        device='GPU', 
        max_batch_size=1, 
        calibrator=None, 
        dla_core=0
        ):
        self.onnxpath=onnxpath
        self.enginepath=onnxpath+f'.{precision}.{device}.{dla_core}.{max_batch_size}.trt'
        #filename to be used for saving and reading engines
        self.nclasses=4
        self.pp_mean=np.array([0.485, 0.456, 0.406]).reshape((1,1,3))
        self.pp_stdev=np.array([0.229, 0.224, 0.225]).reshape((1,1,3))
        #mean and stdev for pre-processing images, see torchvision documentation
        self.colors = colors
        self.lanecolor = lanecolor #colormap for lane
        self.midcolor = midcolor
        self.carmancolor = obstacle
        self.in_w=insize[0]
        self.in_h=insize[1] #width, height of input images
        self.outsize = outsize

        #here we specify very important engine build flags
        self.maxworkspace=maxworkspace
        self.max_batch_size=max_batch_size
        
        self.precision_str=precision
        self.precision={'FP16':0, 'INT8':1, 'FP32': -1}[precision]
        #mapping strings to tensorrt precision flags

        self.device={'GPU':trt.DeviceType.GPU, 'DLA': trt.DeviceType.DLA}[device]
        #mapping strings to tensorrt device types

        self.dla_core=dla_core #used only if DLA device is selected
        self.calibrator=calibrator #used only for INT8 precision
        self.allowGPUFallback=3 #used only if DLA is selected
        
        self.engine, self.logger= self.parse_or_load()
        
        self.context=self.engine.create_execution_context()
        self.trt2np_dtype={'FLOAT':np.float32, 'HALF':np.float16, 'INT8':np.int8}
        self.dtype = self.trt2np_dtype[self.engine.get_binding_dtype(0).name]
        
        self.allocate_buffers(np.zeros((1,3,self.in_h,self.in_w), dtype=self.dtype))

    def allocate_buffers(self, image):
        pass
        insize=image.shape[-2:]
        # outsize=[insize[0] >> 3, insize[1] >> 3]

        self.output=np.empty((self.nclasses,self.outsize[0],self.outsize[1]), dtype=self.dtype)
        self.d_input=cuda.mem_alloc(image.nbytes)
        self.d_output=cuda.mem_alloc(self.output.nbytes)

        self.bindings=[int(self.d_input), int(self.d_output)]
        #print(self.bindings)
        self.stream=cuda.Stream()

    def preprocess(self, img):
        img=cv2.resize(img,(self.in_w,self.in_h))
        img=img[...,::-1]
        img=img.astype(np.float32)/255
        img=(img-self.pp_mean)/self.pp_stdev

        img=np.transpose(img,(2,0,1))
        img=np.ascontiguousarray(img[None,...]).astype(self.dtype)

        # img = self.transforms(img)
        # img = np.ascontiguousarray(img).astype(self.dtype)

        return img

    def infer(self, image, benchmark=False):
        """
        image: unresized,
        """
        intensor=self.preprocess(image)

        start=time.time()

        cuda.memcpy_htod_async(self.d_input, intensor, self.stream)
        self.context.execute_async_v2(self.bindings, self.stream.handle, None)
        cuda.memcpy_dtoh_async(self.output, self.d_output, self.stream)

        self.stream.synchronize()
        
        if benchmark:
            duration=(time.time()-start)
            return duration

    def infer_async(self, intensor):
        #intensor should be preprocessed tensor
        cuda.memcpy_htod_async(self.d_input, intensor, self.stream)
        self.context.execute_async_v2(self.bindings, self.stream.handle, None)
        cuda.memcpy_dtoh_async(self.output, self.d_output, self.stream)

    def draw(self, img):
        shape=(img.shape[1],img.shape[0])
        segres=np.transpose(self.output,(1,2,0)).astype(np.float32)

        segres=cv2.resize(segres, shape)
        mask=segres.argmax(axis=-1)
        colored = self.colors[mask]
        colored_lane=self.lanecolor[mask]
        # colored_mid_lane=self.midcolor[mask]
        colored_obstacle = self.carmancolor[mask]

        drawn=cv2.addWeighted(img, 0.8, colored, 0.5, 0.0)
        # mask_colored = cv2.
        return colored_lane, colored_obstacle, drawn
        # drawn=cv2.addWeighted(img, 0.5, colored, 0.5, 0.0)

    def infervideo(self, src):
        # src=cv2.VideoCapture(infile)
        ret,frame=src.read()
        if not ret:
            print('Cannot read file/camera: {}')

        while ret:
            t = time.time()
            # frame = distort_calib2(frame)
            frame = cv2.resize(frame, (360,360))
            duration=self.infer(frame, benchmark=True)
            _,_,_,drawn=self.draw(frame)
            cv2.imshow('segmented', drawn)
            # cv2.imshow('mask-colored', mask)
            k=cv2.waitKey(1)
            if k==ord('q'):
                break

            print('FPS=:{:.2f}'.format(1/(time.time()-t)))
            ret,frame=src.read()
        return None

    def parse_or_load(self):
        logger= trt.Logger(trt.Logger.INFO)
        #we want to show logs of type info and above (warnings, errors)
        
        if os.path.exists(self.enginepath):
            logger.log(trt.Logger.INFO, 'Found pre-existing engine file')
            with open(self.enginepath, 'rb') as f:
                rt=trt.Runtime(logger)
                engine=rt.deserialize_cuda_engine(f.read())

            return engine, logger

        else: #parse and build if no engine found
            with trt.Builder(logger) as builder:
                builder.max_batch_size=self.max_batch_size
                #setting max_batch_size isn't strictly necessary in this case
                #since the onnx file already has that info, but its a good practice
                
                network_flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
                
                #since the onnx file was exported with an explicit batch dim,
                #we need to tell this to the builder. We do that with EXPLICIT_BATCH flag
                
                with builder.create_network(network_flag) as net:
                
                    with trt.OnnxParser(net, logger) as p:
                        #create onnx parser which will read onnx file and
                        #populate the network object `net`					
                        with open(self.onnxpath, 'rb') as f:
                            if not p.parse(f.read()):
                                for err in range(p.num_errors):
                                    print(p.get_error(err))
                            else:
                                logger.log(trt.Logger.INFO, 'Onnx file parsed successfully')

                        net.get_input(0).dtype=trt.DataType.HALF
                        net.get_output(0).dtype=trt.DataType.HALF
                        #we set the inputs and outputs to be float16 type to enable
                        #maximum fp16 acceleration. Also helps for int8
                        
                        config=builder.create_builder_config()
                        #we specify all the important parameters like precision, 
                        #device type, fallback in config object

                        config.max_workspace_size = self.maxworkspace

                        if self.precision_str in ['FP16', 'INT8']:
                            config.flags = ((1<<self.precision)|(1<<self.allowGPUFallback))
                            config.DLA_core=self.dla_core
                        # DLA core (0 or 1 for Jetson AGX/NX/Orin) to be used must be 
                        # specified at engine build time. An engine built for DLA0 will 
                        # not work on DLA1. As such, to use two DLA engines simultaneously, 
                        # we must build two different engines.

                        config.default_device_type=self.device
                        #if device is set to GPU, DLA_core has no effect

                        config.profiling_verbosity = trt.ProfilingVerbosity.VERBOSE
                        #building with verbose profiling helps debug the engine if there are
                        #errors in inference output. Does not impact throughput.

                        if self.precision_str=='INT8' and self.calibrator is None:
                            logger.log(trt.Logger.ERROR, 'Please provide calibrator')
                            #can't proceed without a calibrator
                            quit()
                        elif self.precision_str=='INT8' and self.calibrator is not None:
                            config.int8_calibrator=self.calibrator
                            logger.log(trt.Logger.INFO, 'Using INT8 calibrator provided by user')

                        logger.log(trt.Logger.INFO, 'Checking if network is supported...')
                        
                        # if builder.is_network_supported(net, config):
                        # 	logger.log(trt.Logger.INFO, 'Network is supported')
                        # 	#tensorRT engine can be built only if all ops in network are supported.
                        # 	#If ops are not supported, build will fail. In this case, consider using 
                        # 	#torch-tensorrt integration. We might do a blog post on this in the future.
                        # else:
                        # 	logger.log(trt.Logger.ERROR, 'Network contains operations that are not supported by TensorRT')
                        # 	logger.log(trt.Logger.ERROR, 'QUITTING because network is not supported')
                        # 	quit()

                        if self.device==trt.DeviceType.DLA:
                            dla_supported=0
                            logger.log(trt.Logger.INFO, 'Number of layers in network: {}'.format(net.num_layers))
                            for idx in range(net.num_layers):
                                if config.can_run_on_DLA(net.get_layer(idx)):
                                    dla_supported+=1

                            logger.log(trt.Logger.INFO, f'{dla_supported} of {net.num_layers} layers are supported on DLA')

                        logger.log(trt.Logger.INFO, 'Building inference engine...')
                        engine=builder.build_engine(net, config)
                        #this will take some time

                        logger.log(trt.Logger.INFO, 'Inference engine built successfully')

                        with open(self.enginepath, 'wb') as s:
                            s.write(engine.serialize())
                        logger.log(trt.Logger.INFO, f'Inference engine saved to {self.enginepath}')
                        
        return engine, 

if __name__ == '__main__':

    device='GPU'
    precision='FP16'
    dla_core=int(device[3:]) if len(device)>3 else 0
    # print(dla_core)
    #checkpoints_trt/bisenet110_360.onnx
    seg=TRTSegmentor('checkpoints_trt/bisenet18_2_7.onnx', colors, lanecolor, midcolor,
        device=device, 
        precision=precision,
        calibrator=None, 
        dla_core=dla_core)
    cap = cv2.VideoCapture('video/project2.avi')
    # cap = cv2.VideoCapture(" v4l2src device=/dev/video1 ! image/jpeg, format=MJPG, width=1280, height=720 ! nvv4l2decoder mjpeg=1 ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink drop=1", cv2.CAP_GSTREAMER)
    seg.infervideo(cap)

    print('Inferred successfully')
