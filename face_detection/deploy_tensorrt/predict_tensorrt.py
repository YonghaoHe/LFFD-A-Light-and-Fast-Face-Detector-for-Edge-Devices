import sys
import os
import logging
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import numpy
import cv2

logging.getLogger().setLevel(logging.INFO)


def NMS(boxes, overlap_threshold):
    '''

    :param boxes: numpy nx5, n is the number of boxes, 0:4->x1, y1, x2, y2, 4->score
    :param overlap_threshold:
    :return:
    '''
    if boxes.shape[0] == 0:
        return boxes

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype != numpy.float32:
        boxes = boxes.astype(numpy.float32)

    # initialize the list of picked indexes
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    sc = boxes[:, 4]
    widths = x2 - x1
    heights = y2 - y1

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = heights * widths
    idxs = numpy.argsort(sc)  # 从小到大排序

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # compare secend highest score boxes
        xx1 = numpy.maximum(x1[i], x1[idxs[:last]])
        yy1 = numpy.maximum(y1[i], y1[idxs[:last]])
        xx2 = numpy.minimum(x2[i], x2[idxs[:last]])
        yy2 = numpy.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bo（ box
        w = numpy.maximum(0, xx2 - xx1 + 1)
        h = numpy.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = numpy.delete(idxs, numpy.concatenate(([last], numpy.where(overlap > overlap_threshold)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick]


# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


class Inference_TensorRT:
    def __init__(self, onnx_file_path,
                 receptive_field_list,
                 receptive_field_stride,
                 bbox_small_list,
                 bbox_large_list,
                 receptive_field_center_start,
                 num_output_scales):

        temp_trt_file = os.path.join('trt_file_cache/', os.path.basename(onnx_file_path).replace('.onnx', '.trt'))

        load_trt_flag = False
        if not os.path.exists(temp_trt_file):
            if not os.path.exists(onnx_file_path):
                logging.error('ONNX file does not exist!')
                sys.exit(1)
            logging.info('Init engine from ONNX file.')
        else:
            load_trt_flag = True
            logging.info('Init engine from serialized engine.')

        self.receptive_field_list = receptive_field_list
        self.receptive_field_stride = receptive_field_stride
        self.bbox_small_list = bbox_small_list
        self.bbox_large_list = bbox_large_list
        self.receptive_field_center_start = receptive_field_center_start
        self.num_output_scales = num_output_scales
        self.constant = [i / 2.0 for i in self.receptive_field_list]

        # init log
        TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
        self.engine = None
        if load_trt_flag:
            with open(temp_trt_file, 'rb') as fin, trt.Runtime(TRT_LOGGER) as runtime:
                self.engine = runtime.deserialize_cuda_engine(fin.read())
        else:
            # declare builder object
            logging.info('Create TensorRT builder.')
            builder = trt.Builder(TRT_LOGGER)

            # get network object via builder
            logging.info('Create TensorRT network.')
            network = builder.create_network()

            # create ONNX parser object
            logging.info('Create TensorRT ONNX parser.')
            parser = trt.OnnxParser(network, TRT_LOGGER)

            with open(onnx_file_path, 'rb') as onnx_fin:
                parser.parse(onnx_fin.read())

            # print possible errors
            num_error = parser.num_errors
            if num_error != 0:
                logging.error('Errors occur while parsing the ONNX file!')
                for i in range(num_error):
                    temp_error = parser.get_error(i)
                    print(temp_error.desc())
                sys.exit(1)

            # create engine via builder
            builder.max_batch_size = 1
            builder.average_find_iterations = 2
            logging.info('Create TensorRT engine...')
            engine = builder.build_cuda_engine(network)

            # serialize engine
            if not os.path.exists('trt_file_cache/'):
                os.makedirs('trt_file_cache/')
            logging.info('Serialize the engine for fast init.')
            with open(os.path.join('trt_file_cache/', os.path.basename(onnx_file_path).replace('.onnx', '.trt')), 'wb') as fout:
                fout.write(engine.serialize())
            self.engine = engine

        self.output_shapes = []
        self.input_shapes = []
        for binding in self.engine:
            if self.engine.binding_is_input(binding):
                self.input_shapes.append(tuple([self.engine.max_batch_size] + list(self.engine.get_binding_shape(binding))))
            else:
                self.output_shapes.append(tuple([self.engine.max_batch_size] + list(self.engine.get_binding_shape(binding))))
        if len(self.input_shapes) != 1:
            logging.error('Only one input data is supported.')
            sys.exit(1)
        self.input_shape = self.input_shapes[0]
        logging.info('The required input size: %d, %d, %d' % (self.input_shape[2], self.input_shape[3], self.input_shape[1]))

        # create executor
        self.executor = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings = self.__allocate_buffers(self.engine)

    def __allocate_buffers(self, engine):
        inputs = []
        outputs = []
        bindings = []
        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(device_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        return inputs, outputs, bindings

    def do_inference(self, image, score_threshold=0.4, top_k=10000, NMS_threshold=0.4, NMS_flag=True, skip_scale_branch_list=[]):

        if image.ndim != 3 or image.shape[2] != 3:
            print('Only RGB images are supported.')
            return None
        input_height = self.input_shape[2]
        input_width = self.input_shape[3]
        if image.shape[0] != input_height or image.shape[1] != input_width:
            logging.info('The size of input image is not %dx%d.\nThe input image will be resized keeping the aspect ratio.' % (input_height, input_width))

        input_batch = numpy.zeros((1, input_height, input_width, self.input_shape[1]), dtype=numpy.float32)
        left_pad = 0
        top_pad = 0
        if image.shape[0] / image.shape[1] > input_height / input_width:
            resize_scale = input_height / image.shape[0]
            input_image = cv2.resize(image, (0, 0), fx=resize_scale, fy=resize_scale)
            left_pad = int((input_width - input_image.shape[1]) / 2)
            input_batch[0, :, left_pad:left_pad + input_image.shape[1], :] = input_image
        else:
            resize_scale = input_width / image.shape[1]
            input_image = cv2.resize(image, (0, 0), fx=resize_scale, fy=resize_scale)
            top_pad = int((input_height - input_image.shape[0]) / 2)
            input_batch[0, top_pad:top_pad + input_image.shape[0], :, :] = input_image

        input_batch = input_batch.transpose([0, 3, 1, 2])
        input_batch = numpy.array(input_batch, dtype=numpy.float32, order='C')
        self.inputs[0].host = input_batch

        [cuda.memcpy_htod(inp.device, inp.host) for inp in self.inputs]
        self.executor.execute(batch_size=self.engine.max_batch_size, bindings=self.bindings)
        [cuda.memcpy_dtoh(output.host, output.device) for output in self.outputs]
        outputs = [out.host for out in self.outputs]
        outputs = [numpy.squeeze(output.reshape(shape)) for output, shape in zip(outputs, self.output_shapes)]

        bbox_collection = []
        for i in range(self.num_output_scales):
            if i in skip_scale_branch_list:
                continue

            score_map = numpy.squeeze(outputs[i * 2])

            # show feature maps-------------------------------
            # score_map_show = score_map * 255
            # score_map_show[score_map_show < 0] = 0
            # score_map_show[score_map_show > 255] = 255
            # cv2.imshow('score_map' + str(i), cv2.resize(score_map_show.astype(dtype=numpy.uint8), (0, 0), fx=2, fy=2))
            # cv2.waitKey()

            bbox_map = numpy.squeeze(outputs[i * 2 + 1])

            RF_center_Xs = numpy.array([self.receptive_field_center_start[i] + self.receptive_field_stride[i] * x for x in range(score_map.shape[1])])
            RF_center_Xs_mat = numpy.tile(RF_center_Xs, [score_map.shape[0], 1])
            RF_center_Ys = numpy.array([self.receptive_field_center_start[i] + self.receptive_field_stride[i] * y for y in range(score_map.shape[0])])
            RF_center_Ys_mat = numpy.tile(RF_center_Ys, [score_map.shape[1], 1]).T

            x_lt_mat = RF_center_Xs_mat - bbox_map[0, :, :] * self.constant[i]
            y_lt_mat = RF_center_Ys_mat - bbox_map[1, :, :] * self.constant[i]
            x_rb_mat = RF_center_Xs_mat - bbox_map[2, :, :] * self.constant[i]
            y_rb_mat = RF_center_Ys_mat - bbox_map[3, :, :] * self.constant[i]

            x_lt_mat = x_lt_mat
            x_lt_mat[x_lt_mat < 0] = 0
            y_lt_mat = y_lt_mat
            y_lt_mat[y_lt_mat < 0] = 0
            x_rb_mat = x_rb_mat
            x_rb_mat[x_rb_mat > input_width] = input_width
            y_rb_mat = y_rb_mat
            y_rb_mat[y_rb_mat > input_height] = input_height

            select_index = numpy.where(score_map > score_threshold)
            for idx in range(select_index[0].size):
                bbox_collection.append((x_lt_mat[select_index[0][idx], select_index[1][idx]] - left_pad,
                                        y_lt_mat[select_index[0][idx], select_index[1][idx]] - top_pad,
                                        x_rb_mat[select_index[0][idx], select_index[1][idx]] - left_pad,
                                        y_rb_mat[select_index[0][idx], select_index[1][idx]] - top_pad,
                                        score_map[select_index[0][idx], select_index[1][idx]]))

        # NMS
        bbox_collection = sorted(bbox_collection, key=lambda item: item[-1], reverse=True)
        if len(bbox_collection) > top_k:
            bbox_collection = bbox_collection[0:top_k]
        bbox_collection_numpy = numpy.array(bbox_collection, dtype=numpy.float32)
        bbox_collection_numpy = bbox_collection_numpy / resize_scale

        if NMS_flag:
            final_bboxes = NMS(bbox_collection_numpy, NMS_threshold)
            final_bboxes_ = []
            for i in range(final_bboxes.shape[0]):
                final_bboxes_.append((final_bboxes[i, 0], final_bboxes[i, 1], final_bboxes[i, 2], final_bboxes[i, 3], final_bboxes[i, 4]))

            return final_bboxes_
        else:
            return bbox_collection_numpy


def run_prediction_folder():
    import sys
    sys.path.append('..')
    from config_farm import configuration_10_320_20L_5scales_v2 as cfg

    debug_folder = './debug_image'
    file_name_list = [file_name for file_name in os.listdir(debug_folder) if file_name.lower().endswith('jpg')]

    onnx_file_path = './onnx_files/v2.onnx'
    myInference = Inference_TensorRT(
        onnx_file_path=onnx_file_path,
        receptive_field_list=cfg.param_receptive_field_list,
        receptive_field_stride=cfg.param_receptive_field_stride,
        bbox_small_list=cfg.param_bbox_small_list,
        bbox_large_list=cfg.param_bbox_large_list,
        receptive_field_center_start=cfg.param_receptive_field_center_start,
        num_output_scales=cfg.param_num_output_scales)

    for file_name in file_name_list:
        im = cv2.imread(os.path.join(debug_folder, file_name))

        bboxes = myInference.do_inference(im, score_threshold=0.6, top_k=1000, NMS_threshold=0.2, NMS_flag=True)
        for bbox in bboxes:
            cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

        if max(im.shape[:2]) > 1440:
            scale = 1440 / max(im.shape[:2])
            im = cv2.resize(im, (0, 0), fx=scale, fy=scale)
        cv2.imshow('im', im)
        cv2.waitKey()


if __name__ == '__main__':
    run_prediction_folder()
