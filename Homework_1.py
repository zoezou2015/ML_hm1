
import sklearn.datasets as load
import sklearn.feature_extraction.text as feature_extraction
import numpy
import sklearn.metrics as metrics
import numpy.random as nr
import time
import matplotlib.pyplot as plt

category = ['atheism', 'sports']
path1 = '/Users/Zoe/Desktop/HW1/data/train'
path2 = '/Users/Zoe/Desktop/HW1/data/test'

def vector_for_input(train_file_path=path1,
                     test_file_path=path2, categories=None):
    train_data = load.load_files(train_file_path, categories=categories, encoding='utf-8', decode_error='ignore')
    test_data = load.load_files(test_file_path, categories=categories, encoding='utf-8', decode_error='ignore')

    # vectorized_normalized = feature_extraction.TfidfVectorizer(min_df=1)
    # train_input_normalized = vectorized_normalized.fit_transform(train_data['data'])
    # test_input_normalized = vectorized_normalized.transform(test_data['data'])

    vectorized = feature_extraction.CountVectorizer(min_df=1)
    train_input = vectorized.fit_transform(train_data['data'])
    test_input = vectorized.transform(test_data['data'])

    return train_input, train_data['target'], test_input, test_data['target']


def perceptron_comparison(n_iter=5, binary_input=False):
    if binary_input:
        x_train, label_train, x_test, label_test = vector_for_input_binary(categories=['sports', 'atheism'])
    else:
        x_train, label_train, x_test, label_test = vector_for_input(categories=['sports', 'atheism'])
    for i in range(0, label_train.shape[0]):
        if label_train[i] == 0:
            label_train[i] = -1
    for i in range(0, label_test.shape[0]):
        if label_test[i] == 0:
            label_test[i] = -1

    sample_number, feature_number = x_train.shape
    weight = numpy.zeros(feature_number)
    u = numpy.zeros(feature_number)
    beta = 0
    bias = 0
    c = 1
    # training
    starting_time = time.time()
    for it in range(1, n_iter+1):
        for each_sample in range(0, sample_number):
        #for each_sample in nr.permutation(numpy.array(range(0, sample_number))):
            input_array = x_train[each_sample].toarray()
            if ((numpy.inner(input_array, weight) + bias) * (label_train[each_sample]))[0] <= 0:
                weight = numpy.add(weight, numpy.multiply(label_train[each_sample], input_array))
                bias = bias + label_train[each_sample]
                u = numpy.add(u, numpy.multiply(label_train[each_sample]*c, input_array))
                beta += label_train[each_sample]*c
                # print 'weight=', weight, 'bias= ', bias, '\n', 'u= ', u, 'beta= ', beta
            c += 1
    average_weight = weight - numpy.multiply(1.0/c, u)
    average_bias = bias - (1.0/c) * beta
    print 'training_time is ', time.time() - starting_time
    #print average_weight, weight, '\n', average_bias, bias

    # testing
    label_train_predict = numpy.zeros(label_train.shape)
    average_label_train_predict = numpy.zeros(label_train.shape)
    label_test_predict = numpy.zeros(label_test.shape)
    average_label_test_predict = numpy.zeros(label_test.shape)

    for i in range(0, label_train.shape[0]):
        if (numpy.inner(weight, x_train[i].toarray()) + bias) >= 0:
            label_train_predict[i] = 1
        else:
            label_train_predict[i] = -1

    for i in range(0, label_train.shape[0]):
        if (numpy.inner(average_weight, x_train[i].toarray()) + average_bias) >= 0:
            average_label_train_predict[i] = 1
        else:
            average_label_train_predict[i] = -1

    for i in range(0, label_test.shape[0]):
        if (numpy.inner(weight, x_test[i].toarray()) + bias) >= 0:
            label_test_predict[i] = 1
        else:
            label_test_predict[i] = -1
    for i in range(0, label_test.shape[0]):
        if (numpy.inner(average_weight, x_test[i].toarray()) + average_bias) >= 0:
            average_label_test_predict[i] = 1
        else:
            average_label_test_predict[i] = -1
    #print label_test_predict
    #print label_test

    accuracy_test = metrics.accuracy_score(label_test, label_test_predict)
    average_accuracy_test = metrics.accuracy_score(label_test, average_label_test_predict)
    print 'normal perceptron accuracy for testing is', accuracy_test
    print 'average perceptron accuracy for testing is', average_accuracy_test

    accuracy_train = metrics.accuracy_score(label_train, label_train_predict)
    average_accuracy_train = metrics.accuracy_score(label_train, average_label_train_predict)
    print 'normal perceptron accuracy for training is', accuracy_train
    print 'average perceptron accuracy for training is', average_accuracy_train

    return accuracy_test, average_accuracy_test, accuracy_train, average_accuracy_train


def multi_class_perceptron(class_number=4, n_iter=5):
    x_train, label_train, x_test, label_test = vector_for_input()

    sample_number, feature_number = x_train.shape
    weight = numpy.zeros((class_number, feature_number))
    bias = numpy.zeros(class_number)
    start_time = time.time()
    for each_class in range(0, class_number):
        class_count = 0
        label_train_temp = numpy.zeros(label_train.shape)
        for i in range(0, label_train.shape[0]):
            if label_train[i] == each_class:
                label_train_temp[i] = -1
                class_count += 1
            else:
                label_train_temp[i] = 1
        print 'there are total ', class_count, ' ', each_class, 'samples'
        print sample_number - class_count
        # training
        update_times = 0
        each_starting_time = time.time()
        for iter in range(1, n_iter+1):
            for each_sample in nr.permutation(numpy.array(range(0, sample_number))):
                input_array = x_train[each_sample].toarray()
                if ((numpy.inner(input_array, weight[each_class]) + bias[each_class]) * (label_train_temp[each_sample]))[0] <= 0:
                    weight[each_class] = numpy.add(weight[each_class], numpy.multiply(label_train_temp[each_sample],
                                                                                      input_array))
                    bias[each_class] = bias[each_class] + label_train_temp[each_sample]
                    update_times += 1
        print 'training_time for class', each_class, ' is ', time.time() - each_starting_time
        # print weight[each_class], '\n', bias[each_class]
        print 'update ', update_times, ' times'
    print 'total training time is ', time.time() - start_time
    print weight
    print bias
    # testing
    label_train_predict = numpy.ones(label_train.shape)
    label_test_predict = numpy.ones(label_test.shape)

    for i in range(0, label_train.shape[0]):
        predict = numpy.dot(weight, numpy.transpose(x_train[i].toarray())) + bias.reshape(4, 1)
        min_pos = numpy.argmin(predict)
        label_train_predict[i] = min_pos

    for i in range(0, label_test.shape[0]):
        predict = numpy.dot(weight, numpy.transpose(x_test[i].toarray())) + bias.reshape(4, 1)
        min_pos = numpy.argmin(predict)
        label_test_predict[i] = min_pos

    accuracy_test = metrics.accuracy_score(label_test, label_test_predict)
    print 'normal perceptron accuracy for testing is', accuracy_test

    accuracy_train = metrics.accuracy_score(label_train, label_train_predict)
    print 'normal perceptron accuracy for training is', accuracy_train


def hinge_loss(learning_rate=0.1, alpha=0.05):
    x_train, label_train, x_test, label_test = vector_for_input(categories=['sports', 'atheism'])
    for i in range(0, label_train.shape[0]):
        if label_train[i] == 0:
            label_train[i] = -1
    for i in range(0, label_test.shape[0]):
        if label_test[i] == 0:
            label_test[i] = -1

    sample_number, feature_number = x_train.shape
    weight = numpy.zeros(feature_number)
    bias = 0
    c = 0
    # training
    starting_time = time.time()
    loss_updated = 0
    loss_previous = 0
    loss_diff = alpha+1
    n = 0
    while abs(loss_diff) > alpha:
        specific_x = x_train[0].toarray()
        specific_y = 0
        # for each_sample in nr.permutation(numpy.array(range(0, sample_number))):
        for each_sample in nr.choice(numpy.array(range(0, sample_number)), size=1000):
            input_array = x_train[each_sample].toarray()
            # print ((numpy.inner(input_array, weight) + bias) * (label_train[each_sample]))[0]
            if ((numpy.inner(input_array, weight) + bias) * (label_train[each_sample]))[0] < 1:
                # loss_previous = 1 - ((numpy.inner(input_array, weight) + bias) * (label_train[each_sample]))[0]
                weight = numpy.add(weight, numpy.multiply(learning_rate*label_train[each_sample], input_array))
                bias += learning_rate * label_train[each_sample]
                c += 1
                specific_x = input_array
                specific_y = each_sample
        n += 1
        loss_previous = loss_updated
        loss_updated = 1 - ((numpy.inner(specific_x, weight) + bias) * (label_train[specific_y]))[0]
        loss_diff = abs(loss_updated - loss_previous)
        print 'loss_diff at ', 1000 * n, 'points: ', loss_diff

    training_time = time.time() - starting_time
    print loss_previous
    print loss_updated
    print 'updating ', c, ' times'
    print 'training_time is ', training_time

    # testing
    label_train_predict = numpy.ones(label_train.shape)
    label_test_predict = numpy.ones(label_test.shape)

    for i in range(0, label_train.shape[0]):
        if (numpy.inner(weight, x_train[i].toarray()) + bias) >= 0:
            label_train_predict[i] = 1
        else:
            label_train_predict[i] = -1

    for i in range(0, label_test.shape[0]):
        if (numpy.inner(weight, x_test[i].toarray()) + bias) >= 0:
            label_test_predict[i] = 1
        else:
            label_test_predict[i] = -1

    accuracy_test = metrics.accuracy_score(label_test, label_test_predict)
    print 'accuracy for testing is', accuracy_test

    accuracy_train = metrics.accuracy_score(label_train, label_train_predict)
    print 'accuracy for training is', accuracy_train

    return accuracy_test, accuracy_train, training_time


def hinge_loss_regularized(regularization=0.5, learning_rate=0.5, alpha=1.0):
    x_train, label_train, x_test, label_test = vector_for_input(categories=['sports', 'atheism'])
    for i in range(0, label_train.shape[0]):
        if label_train[i] == 0:
            label_train[i] = -1
    for i in range(0, label_test.shape[0]):
        if label_test[i] == 0:
            label_test[i] = -1

    sample_number, feature_number = x_train.shape
    weight = numpy.zeros(feature_number)
    bias = 0
    c = 0
    # training
    starting_time = time.time()
    loss_updated = 0
    loss_previous = 0
    loss_diff = alpha+1
    n = 0
    while abs(loss_diff) > alpha:
        specific_x = x_train[0].toarray()
        specific_y = 0
        # for each_sample in nr.permutation(numpy.array(range(0, sample_number))):
        for each_sample in nr.choice(numpy.array(range(0, sample_number)), size=500):
            input_array = x_train[each_sample].toarray()
            # print ((numpy.inner(input_array, weight) + bias) * (label_train[each_sample]))[0]
            if ((numpy.inner(input_array, weight) + bias) * (label_train[each_sample]))[0] < 1:
                # loss_previous = 1 - ((numpy.inner(input_array, weight) + bias) * (label_train[each_sample]))[0]
                weight = numpy.add(weight * (1 - 2*regularization*learning_rate), numpy.multiply(
                    learning_rate*label_train[each_sample], input_array))
                bias += learning_rate * label_train[each_sample]
                # loss_updated = regularization * numpy.inner(weight, weight) + 1 - (
                #    (numpy.inner(input_array, weight) + bias) * (label_train[each_sample]))[0]
                c += 1
                specific_x = input_array
                specific_y = each_sample
        n += 1
        loss_previous = loss_updated
        loss_updated = regularization * numpy.inner(weight, weight) + 1 - (
            (numpy.inner(specific_x, weight) + bias) * (label_train[specific_y]))[0]
        loss_diff = abs(loss_updated - loss_previous)
        print 'loss_diff at ', 1000 * n, 'points: ', loss_diff

    print loss_previous
    print loss_updated
    print 'updating ', c, ' times'
    print 'training_time is ', time.time() - starting_time
    # print weight, '\n', bias

    # testing
    label_train_predict = numpy.ones(label_train.shape)
    label_test_predict = numpy.ones(label_test.shape)

    for i in range(0, label_train.shape[0]):
        if (numpy.inner(weight, x_train[i].toarray()) + bias) >= 0:
            label_train_predict[i] = 1
        else:
            label_train_predict[i] = -1

    for i in range(0, label_test.shape[0]):
        if (numpy.inner(weight, x_test[i].toarray()) + bias) >= 0:
            label_test_predict[i] = 1
        else:
            label_test_predict[i] = -1

    accuracy_test = metrics.accuracy_score(label_test, label_test_predict)
    print 'accuracy for testing is', accuracy_test

    accuracy_train = metrics.accuracy_score(label_train, label_train_predict)
    print 'accuracy for training is', accuracy_train

    return accuracy_test, accuracy_train


def logistic_regression(learning_rate=0.5, alpha=1.0):
    x_train, label_train, x_test, label_test = vector_for_input(categories=['sports', 'atheism'])
    for i in range(0, label_train.shape[0]):
        if label_train[i] == 0:
            label_train[i] = -1
    for i in range(0, label_test.shape[0]):
        if label_test[i] == 0:
            label_test[i] = -1

    sample_number, feature_number = x_train.shape
    weight = numpy.zeros(feature_number)
    bias = 0
    c = 0
    # training
    starting_time = time.time()
    loss_updated = 0
    loss_previous = 0
    loss_diff = alpha+1
    n = 0
    while abs(loss_diff) > alpha:
        specific_x = x_train[0].toarray()
        specific_y = 0

        # for each_sample in nr.permutation(numpy.array(range(0, sample_number))):
        for each_sample in nr.choice(numpy.array(range(0, sample_number)), size=1000):
            input_array = x_train[each_sample].toarray()
            # print ((numpy.inner(input_array, weight) + bias) * (label_train[each_sample]))[0]
            e = -((numpy.inner(input_array, weight) + bias) * (label_train[each_sample]))[0]
            # print e
            denominator = 1.0/(numpy.exp(e) + 1)
            # loss_previous = 1 - ((numpy.inner(input_array, weight) + bias) * (label_train[each_sample]))[0]
            weight = numpy.add(weight, numpy.multiply(denominator*learning_rate*label_train[each_sample], input_array))
            bias += learning_rate * label_train[each_sample] * denominator
            # loss_updated = regularization * numpy.inner(weight, weight) + 1 - (
            #    (numpy.inner(input_array, weight) + bias) * (label_train[each_sample]))[0]
            c += 1
            specific_x = input_array
            specific_y = each_sample
        n += 1
        loss_previous = loss_updated
        loss_updated = numpy.log(1 + numpy.exp(((numpy.inner(specific_x, weight) + bias) * (label_train[specific_y]))[0]))
        loss_diff = abs(loss_updated - loss_previous)
        print 'loss_diff at ', 1000 * n, 'points: ', loss_diff

    print loss_previous
    print loss_updated
    print 'updating ', c, ' times'
    print 'training_time is ', time.time() - starting_time
    print weight, '\n', bias

    # testing
    label_train_predict = numpy.ones(label_train.shape)
    label_test_predict = numpy.ones(label_test.shape)

    for i in range(0, label_train.shape[0]):
        if (numpy.inner(weight, x_train[i].toarray()) + bias) < 0:
            label_train_predict[i] = 1
        else:
            label_train_predict[i] = -1

    for i in range(0, label_test.shape[0]):
        if (numpy.inner(weight, x_test[i].toarray()) + bias) < 0:
            label_test_predict[i] = 1
        else:
            label_test_predict[i] = -1

    accuracy_test = metrics.accuracy_score(label_test, label_test_predict)
    print 'accuracy for testing is', accuracy_test

    accuracy_train = metrics.accuracy_score(label_train, label_train_predict)
    print 'accuracy for training is', accuracy_train


def vector_for_input_binary(train_file_path="/mnt/hgfs/temp/machine learning/train",
                            test_file_path="/mnt/hgfs/temp/machine learning/test", categories=None):
    train_data = load.load_files(train_file_path, categories=categories, encoding='utf-8', decode_error='ignore')
    test_data = load.load_files(test_file_path, categories=categories, encoding='utf-8', decode_error='ignore')

    vectorized = feature_extraction.CountVectorizer(min_df=1, binary=True)
    train_input = vectorized.fit_transform(train_data['data'])
    test_input = vectorized.transform(test_data['data'])

    return train_input, train_data['target'], test_input, test_data['target']


def question_2(n_iter=10):
    accuracy_test = []
    average_accuracy_test = []
    accuracy_train = []
    average_accuracy_train = []
    times = []
    for i in range(1, n_iter+1):
        print '***********', i, '*************'
        test, average_test, train, average_train = perceptron_comparison(n_iter=i)
        accuracy_test.append(test)
        average_accuracy_test.append(average_test)
        accuracy_train.append(train)
        average_accuracy_train.append(average_train)
        times.append(i)
        print '**********************************'

    p1, = plt.plot(times, accuracy_train, 'k')
    p2, = plt.plot(times, accuracy_test, 'r')
    p3, = plt.plot(times, average_accuracy_train, 'g')
    p4, = plt.plot(times, average_accuracy_test, 'b')
    plt.title('Performance over epochs')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend((p1, p2, p3, p4),
               ('train accuracy', 'test accuracy', 'average train accuracy', 'average test accuracy'),
               loc='best', frameon=True)
    plt.savefig('figure_.png', format='png')
    plt.show()


def question_3():
    accuracy_train = []
    accuracy_test = []
    alpha = [0, 0.001, 0.01, 0.05, 0.1, 0.5, 0.8, 1, 5, 10, 20, 30, 50]
    for al in alpha:
        print '*******************'
        test, train, t_time = hinge_loss(learning_rate=0.5, alpha=al)
        print '********************'
        accuracy_test.append(test)
        accuracy_train.append(train)

    p1, = plt.plot(alpha, accuracy_train, 'k')
    p2, = plt.plot(alpha, accuracy_test, 'r')
    plt.title('accuracy over convergence condition')
    plt.xlabel('convergence condition')
    plt.ylabel('accuracy')
    plt.legend((p1, p2),
               ('train accuracy', 'test accuracy'),
               loc='best', frameon=True)
    plt.savefig('figure_4_2.png', format='png')
    plt.show()

    accuracy_test = []
    accuracy_train = []
    rates = []
    training_time = []
    for rate in numpy.arange(0, 1.0, 0.05):
        print '*******************'
        test, train, t_time = hinge_loss(learning_rate=rate, alpha=0.001)
        print '********************'
        accuracy_test.append(test)
        accuracy_train.append(train)
        rates.append(rate)
        training_time.append(t_time)

    plt.plot(rates, training_time, 'r')
    plt.title('training time over learning rate')
    plt.xlabel('learning rate')
    plt.ylabel('training time')
    plt.savefig('figure_2_1.png', format='png')
    plt.show()

    p1, = plt.plot(rates, accuracy_train, 'k')
    p2, = plt.plot(rates, accuracy_test, 'r')
    plt.title('testing accuracy over learning rate')
    plt.xlabel('learning rate')
    plt.ylabel('testing accuracy')
    plt.legend((p1, p2),
               ('train accuracy', 'test accuracy'),
               loc='best', frameon=True)
    plt.savefig('figure_3_1.png', format='png')
    plt.show()


def question_4():
    multi_class_perceptron()


def question_5():
    accuracy_test = []
    accuracy_train = []
    alpha = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]
    for al in alpha:
        print '*******************'
        test, train = hinge_loss_regularized(regularization=al, learning_rate=0.5, alpha=10)
        print '********************'
        accuracy_test.append(test)
        accuracy_train.append(train)

    p1, = plt.plot(alpha, accuracy_train, 'k')
    p2, = plt.plot(alpha, accuracy_test, 'r')
    plt.title('accuracy over regularization constant')
    plt.xlabel('regularization constant')
    plt.ylabel('accuracy')
    plt.ylim((0, 1))
    plt.legend((p1, p2),
               ('train accuracy', 'test accuracy'),
               loc='best', frameon=True)
    plt.savefig('figure_5.png', format='png')
    plt.show()


def question_6():
    perceptron_comparison(binary_input=True)

    perceptron_comparison()

logistic_regression(learning_rate=0.5, alpha=1.0)
question_2()
