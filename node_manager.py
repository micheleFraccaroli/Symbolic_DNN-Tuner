import tensorflow as tf


def get_next_node(node: tf.compat.v1.profiler.MultiGraphNodeProto):
    return node.children[0]


def get_name(node):
    return node.name


def get_flops(node):
    return node.float_ops


def get_next(node):
    node = get_next_node(node)
    return get_name(node), get_flops(node)


def to_dict(node):
    if len(node.children) > 0:
        node = get_next_node(node)
        dict = to_dict(node)
        dict[get_name(node)] = get_flops(node)
        return dict
    else:
        return {}
