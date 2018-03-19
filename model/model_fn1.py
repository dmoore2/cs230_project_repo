"""Define the model."""

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, Flatten,MaxPooling2D



def build_model(is_training, inputs, params):
    """Compute logits of the model (output distribution)

    Args:
        is_training: (bool) whether we are training or not
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) hyperparameters

    Returns:
        output: (tf.Tensor) output of the model
    """
    images = inputs['images'] 
    
    # images.get_shape().as_list() should equal 
    # [batch_size,image_height,image_width, layers per image (IE: rgb = 3, rgbd = 4), num of camera angles]
    
    
    # Ours is hard coded as [batch_size,299,299,12]
    
    #CHANGE IF HARD CODED DIFF
    print(images.get_shape().as_list())
    assert images.get_shape().as_list() == [None, params.image_size, params.image_size, 12]
    

    out = images #[batch_size,299,299,12]
    
    #X = tf.Variable(tf.placeholder(tf.int32, shape=(None,width,height,env_rgbd_cam)))
                                   
    C1,C2,C3,C4 = tf.split(out,[3,3,3,3],3) #A1.get_shape() = (None,width,height,3,1)
    C = [C1,C2,C3,C4]
    print("entering enumerate")
    #Apply CNN to each of 4 images
    for idx, V in enumerate(C):
        #V = tf.squeeze(V) #V.get_shape(): (None,width,height,3,1) -> (None,width,height,3)
        
                                                    
        print(V.get_shape().as_list())
        V = Conv2D(32, input_shape = (V.get_shape()), activation='relu',kernel_size=(7,7), strides=(1, 1))(V)
        V = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(V)
        
        
                                                    
        C[idx] = V
        
   
    #A1.get_shape() = (None,Weight,Height,Layers
    X = tf.concat(C,3) 
    #X.get_shape() = (None,W,H,3*4)
    
                                                
                                    
    X = Conv2D(16, input_shape = (X.get_shape()), activation='relu',kernel_size=(3,3), strides=(1, 1))(V)
    X = MaxPooling2D(pool_size=(4, 4), strides=(4, 4))(X)
    
    X = Flatten()(X)
    X = Dense(55,activation = 'sigmoid')(X)
    
                                                
    return X
    
        
    #X = Conv2D(55, input_shape = (None, 299, 299, 32), activation='relu',kernel_size=(5, 5), strides=(1, 1))(out)
    #X = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(X)
    #X = Flatten()(X)
    #X = Dense(55,activation = 'relu')(X)
    #return X

    #Split into view 1, view 2, view 3, view 4.
    #A = tf.split(out,[1,1,1,1],3)(out) #A[0] is camera 1 rgbd.    A[0].size = [batch_size,299,299,4]
    #assert A.get_shape().as_list() == [None, params.image_size, params.image_size, 4]
    #width = images.get_shape().as_list()[1]
    #height = images.get_shape().as_list()[2]
    #env_rgbd_cam = images.get_shape().as_list()[3]*images.get_shape().as_list()[4]
    
    '''
    C4 has 16 kernels of size 3 × 3 × 3c with stride of 1 pixel. S2 pools the merged features with a stride of 4. Both C5 and C6 have 32 kernels with size of 3 × 3 × 3 with stride of 1 pixel. The dropout is applied to the output of S4 which has been flattened. The fully connected layer FC1 has 32 neurons and FC2 has 3 neurons. The activation of the output layer is softmax function.
    '''
   
    
    return X
    # Define the number of channels of each convolution
    # For each block, we do: 3x3 conv -> batch norm -> relu -> 2x2 maxpool
    num_channels = params.num_channels
    bn_momentum = params.bn_momentum
    channels = [num_channels, num_channels * 2, num_channels * 4, num_channels * 8]
    for i, c in enumerate(channels):
        with tf.variable_scope('block_{}'.format(i+1)):
            out = tf.layers.conv2d(out, c, 3, padding='same')
            if params.use_batch_norm:
                out = tf.layers.batch_normalization(out, momentum=bn_momentum, training=is_training)
            out = tf.nn.relu(out)
            out = tf.layers.max_pooling2d(out, 2, 2)
    print(out.get_shape().as_list())
    #assert out.get_shape().as_list() == [None, 18, 18, num_channels * 8]

    out = tf.reshape(out, [-1, 18 * 18 * num_channels * 8])
    with tf.variable_scope('fc_1'):
        out = tf.layers.dense(out, num_channels * 8)
        if params.use_batch_norm:
            out = tf.layers.batch_normalization(out, momentum=bn_momentum, training=is_training)
        out = tf.nn.relu(out)
    with tf.variable_scope('fc_2'):
        logits = tf.layers.dense(out, params.num_labels)
        print("nums labels")
        print(params.num_labels)
        #logits = tf.nn.sigmoid(logits)

    return logits


def model_fn(mode, inputs, params, reuse=False):
    """Model function defining the graph operations.

    Args:
        mode: (string) can be 'train' or 'eval'
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)
        reuse: (bool) whether to reuse the weights

    Returns:
        model_spec: (dict) contains the graph operations or nodes needed for training / evaluation
    """
    is_training = (mode == 'train')
    labels = inputs['labels']
    labels = tf.cast(labels, tf.int64)

    # -----------------------------------------------------------
    # MODEL: define the layers of the model
    with tf.variable_scope('model', reuse=reuse):
        # Compute the output distribution of the model and the predictions
        logits = build_model(is_training, inputs, params)
        predictions = tf.cast(tf.round((tf.nn.sigmoid(logits))),tf.int64)
       

    # Define loss and accuracy
    print(labels)
    print(logits)
    #loss = tf.losses.sigmoid_cross_entropy(multi_class_laxbels=labels, logits=logits)
    
    #loss = tf.nn.weighted_cross_entropy_with_logits(targets = labels,logits = logits,pos_weights tf.divide(tf.reduce_sum(ones(tf.size(labels)),axis =1),tf.reduce_sum(labels,axis=1)))
    #logits = tf.multiply(logits, tf.cast(labels,tf.float32)) # 
    loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=labels, logits=logits)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(labels,tf.int64), predictions), tf.float32))

    # Define training step that minimizes the loss with the Adam optimizer
    if is_training:
        optimizer = tf.train.AdamOptimizer(params.learning_rate)
        global_step = tf.train.get_or_create_global_step()
        if params.use_batch_norm:
            # Add a dependency to update the moving mean and variance for batch normalization
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                train_op = optimizer.minimize(loss, global_step=global_step)
        else:
            train_op = optimizer.minimize(loss, global_step=global_step)


    # -----------------------------------------------------------
    # METRICS AND SUMMARIES
    # Metrics for evaluation using tf.metrics (average over whole dataset)
    with tf.variable_scope("metrics"):
        metrics = {
            'accuracy': tf.metrics.accuracy(labels=labels, predictions=tf.cast(tf.round((tf.nn.sigmoid(logits))),tf.int64)),
            'recall': tf.metrics.recall(labels=labels, predictions=tf.cast(tf.round((tf.nn.sigmoid(logits))),tf.int64)),
            'loss': tf.metrics.mean(loss)
        }

    # Group the update ops for the tf.metrics
    update_metrics_op = tf.group(*[op for _, op in metrics.values()])

    # Get the op to reset the local variables used in tf.metrics
    metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
    metrics_init_op = tf.variables_initializer(metric_variables)

    # Summaries for training
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)
    #tf.summary.image('train_image', inputs['images'])

    #TODO: if mode == 'eval': ?
    # Add incorrectly labeled images
    mask = tf.not_equal(labels, predictions)
    print(mask)
    # Add a different summary to know how they were misclassified
    #for label in range(0, params.num_labels):
        #mask_label = tf.logical_and(mask, tf.equal(predictions, label))
        #incorrect_image_label = tf.boolean_mask(inputs['labels'][i], mask_label)
        #tf.summary.image('incorrectly_labeled_{}'.format(label), incorrect_image_label)

    # -----------------------------------------------------------
    # MODEL SPECIFICATION
    # Create the model specification and return it
    # It contains nodes or operations in the graph that will be used for training and evaluation
    model_spec = inputs
    model_spec['variable_init_op'] = tf.global_variables_initializer()
    model_spec["predictions"] = predictions
    model_spec['loss'] = loss
    model_spec['accuracy'] = accuracy
    model_spec['metrics_init_op'] = metrics_init_op
    model_spec['metrics'] = metrics
    model_spec['update_metrics'] = update_metrics_op
    model_spec['summary_op'] = tf.summary.merge_all()

    if is_training:
        model_spec['train_op'] = train_op

    return model_spec
