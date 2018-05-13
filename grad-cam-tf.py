from tensorflow_vgg import vgg16, utils
from tensorflow.python.framework import ops
import tensorflow as tf
import numpy as np
import sys
import os
import cv2
import argparse

def register_gradient():
    """
    register gradients for ReLU
    """
    if "GuidedBackPropReLU" not in ops._gradient_registry._registry:
        @tf.RegisterGradient("GuidedBackPropReLU")
        def _GuidedBackPropReLU(op, grad):
            dtype = op.inputs[0].dtype
            return grad * tf.cast(grad > 0., dtype) * tf.cast(op.inputs[0] > 0., dtype)

    if "DeconvReLU" not in ops._gradient_registry._registry:
        @ops.RegisterGradient("DeconvReLU")
        def _DeconvReLU(op, grad):
            dtype = op.inputs[0].dtype
            return grad * tf.cast(grad > 0., dtype)


def everride_relu(name, images, path_vgg):
    """
    override ReLU for guided backpropagation
    """
    g = tf.get_default_graph()
    with g.gradient_override_map({'Relu': name}):
        model = vgg16.Vgg16(vgg16_npy_path=path_vgg)
        with tf.name_scope("content_vgg_gbp"):
            model.build(images)
    return model


def saliency_by_class(input_model, images, category_index, nb_classes=1000):
    """
    calculate d prob_of_class / d image
    """
    loss = tf.multiply(input_model.prob, tf.one_hot([category_index], nb_classes))
    reduced_loss = tf.reduce_sum(loss, axis=1)
    grads = tf.gradients(reduced_loss, images)
    return grads


def grad_cam(input_model,  category_index, layer_name, sess, feed_dict, nb_classes = 1000):
    """
    calculate Grad-CAM
    """
    loss = tf.multiply(input_model.prob, tf.one_hot([category_index], nb_classes))
    reduced_loss = tf.reduce_sum(loss[0])
    conv_output = sess.graph.get_tensor_by_name(layer_name + ':0')
    grads = tf.gradients(reduced_loss, conv_output)[0] # d loss / d conv
    output, grads_val = sess.run([conv_output, grads], feed_dict=feed_dict)
    weights = np.mean(grads_val, axis=(1, 2)) # average pooling
    cams = np.sum(weights * output, axis=3)
    return cams


def save_cam(cams, rank, class_id, class_name, prob, image_batch, input_image_path):
    """
    save Grad-CAM images
    """
    cam = cams[0] # the first GRAD-CAM for the first image in  batch
    image = np.uint8(image_batch[0][:, :, ::-1] * 255.0) # RGB -> BGR
    cam = cv2.resize(cam, (224, 224)) # enlarge heatmap
    cam = np.maximum(cam, 0)
    heatmap = cam / np.max(cam) # normalize
    cam = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET) # balck-and-white to color
    cam = np.float32(cam) + np.float32(image) # everlay heatmap onto the image
    cam = 255 * cam / np.max(cam)
    cam = np.uint8(cam)

    # create image file names
    base_path, ext = os.path.splitext(input_image_path)
    base_path_class = "{}_{}_{}_{}_{:.3f}".format(base_path, rank, class_id, class_name, prob)
    cam_path = "{}_{}{}".format(base_path_class, "gradcam", ext)
    heatmap_path = "{}_{}{}".format(base_path_class, "heatmap", ext)
    segmentation_path = "{}_{}{}".format(base_path_class, "segmented", ext)

    # write images
    cv2.imwrite(cam_path, cam)
    cv2.imwrite(heatmap_path, (heatmap * 255.0).astype(np.uint8))
    cv2.imwrite(segmentation_path, (heatmap[:, :, None].astype(float) * image).astype(np.uint8))


def get_info(prob, file_path, top_n=5):
    """
    returns top_n information(class id, class name, probability and synset data
    """
    synsets = [l.strip() for l in open(file_path).readlines()]
    preds = np.argsort(prob)[::-1]

    top_n_synset = []
    for i in range(top_n):
        pred = preds[i]
        synset = synsets[pred]
        class_name = "_".join(synset.split(",")[0].split(" ")[1:])
        top_n_synset.append( (pred, class_name, prob[pred], synset) )
    return top_n_synset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_image', type=str, default='/path/to/image', help='path to image.')
    parser.add_argument('vgg16_path', type=str, default='/path/to/vgg16.npy', help='path to vgg16.npy.')
    parser.add_argument('--top_n', type=int, default=3, help="Grad-CAM for top N predicted classes.")
    args = parser.parse_args()
    print(args)

    input_image = utils.load_image(args.input_image) # tf RGB
    image_batch = input_image[None, :, :, :3]

    graph = tf.Graph()
    sess = tf.InteractiveSession(graph=graph)
    with tf.device('/cpu:0'):
        images = tf.placeholder("float", [None, 224, 224, 3])
        model = vgg16.Vgg16(vgg16_npy_path=args.vgg16_path)
        with tf.name_scope("content_vgg"):
            model.build(images)

    path_synset = os.path.join(os.path.dirname(vgg16.__file__), "synset.txt")
    prob = sess.run(model.prob, feed_dict={images: image_batch})
    infos = get_info(prob[0], path_synset, top_n=args.top_n)
    for rank, info in enumerate(infos):
        print("{}: class id: {}, class name: {}, probability: {:.3f}, synset: {}".format(rank, *info))

    # GRAD-CAM
    for i in range(args.top_n):
        class_id = infos[i][0]
        class_name = infos[i][1]
        prob = infos[i][2]
        cams = grad_cam(model, class_id, "content_vgg/conv5_3/Relu", sess, feed_dict={images: image_batch})

        save_cam(cams, i, class_id, class_name, prob, image_batch, args.input_image)

    # Guided Backpropagation
    register_gradient()

    del model
    images = tf.placeholder("float", [None, 224, 224, 3])

    guided_model = everride_relu('GuidedBackPropReLU', images, args.vgg16_path)
    class_id = infos[0][0]
    class_saliencies = saliency_by_class(guided_model, images, class_id, nb_classes=1000)
    class_saliency = sess.run(class_saliencies, feed_dict={images: image_batch})[0][0]

    class_saliency = class_saliency - class_saliency.min()
    class_saliency = class_saliency / class_saliency.max() * 255.0
    base_path, ext = os.path.splitext(args.input_image)
    gbprop_path = "{}_{}{}".format(base_path, "guided_bprop", ext)
    cv2.imwrite(gbprop_path, class_saliency.astype(np.uint8))

if __name__ == '__main__':
    main()