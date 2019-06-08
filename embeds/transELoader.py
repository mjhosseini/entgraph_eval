# coding:utf-8
import numpy as np
import tensorflow as tf
import os
import time
import datetime
import ctypes
import sys

ll = ctypes.cdll.LoadLibrary
lib = ll("./init.so")
test_lib = ll("./test.so")
np.set_printoptions(threshold='nan')
np.set_printoptions(linewidth=np.nan)


class Config(object):
    def __init__(self, modelName, relsFolder):
        # lib.setInPath("./data/FB15K/")
        # test_lib.setInPath("./data/FB15K/")
        lib.setInPath("./data/"+relsFolder+"/")  # TODO: be careful
        test_lib.setInPath("./data/"+relsFolder+"/")
        self.modelName = modelName
        lib.setBernFlag(0)
        self.learning_rate = 0.001
        self.testFlag = False
        self.loadFromData = True  # TODO: be careful
        self.L1_flag = True
        self.hidden_size = 300
        self.nbatches = 100
        self.entity = 0
        self.relation = 0
        self.trainTimes = 1000  # TODO: be careful
        self.margin = 1.0


class TransEModel(object):
    def __init__(self, config):

        entity_total = config.entity
        relation_total = config.relation
        batch_size = config.batch_size
        size = config.hidden_size
        margin = config.margin

        self.pos_h = tf.placeholder(tf.int32, [None])
        self.pos_t = tf.placeholder(tf.int32, [None])
        self.pos_r = tf.placeholder(tf.int32, [None])

        self.neg_h = tf.placeholder(tf.int32, [None])
        self.neg_t = tf.placeholder(tf.int32, [None])
        self.neg_r = tf.placeholder(tf.int32, [None])

        with tf.name_scope("embedding"):
            self.ent_embeddings = tf.get_variable(name="ent_embedding", shape=[entity_total, size],
                                                  initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            self.rel_embeddings = tf.get_variable(name="rel_embedding", shape=[relation_total, size],
                                                  initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            pos_h_e = tf.nn.embedding_lookup(self.ent_embeddings, self.pos_h)
            pos_t_e = tf.nn.embedding_lookup(self.ent_embeddings, self.pos_t)
            pos_r_e = tf.nn.embedding_lookup(self.rel_embeddings, self.pos_r)
            neg_h_e = tf.nn.embedding_lookup(self.ent_embeddings, self.neg_h)
            neg_t_e = tf.nn.embedding_lookup(self.ent_embeddings, self.neg_t)
            neg_r_e = tf.nn.embedding_lookup(self.rel_embeddings, self.neg_r)

        if config.L1_flag:
            pos = tf.reduce_sum(abs(pos_h_e + pos_r_e - pos_t_e), 1, keep_dims=True)
            neg = tf.reduce_sum(abs(neg_h_e + neg_r_e - neg_t_e), 1, keep_dims=True)
            self.predict = pos
        else:
            pos = tf.reduce_sum((pos_h_e + pos_r_e - pos_t_e) ** 2, 1, keep_dims=True)
            neg = tf.reduce_sum((neg_h_e + neg_r_e - neg_t_e) ** 2, 1, keep_dims=True)
            self.predict = pos

        with tf.name_scope("output"):
            self.loss = tf.reduce_sum(tf.maximum(pos - neg + margin, 0))


def getRelatoinEmbeddings(modelName, relsFolder):
    print ("loading transE embeddings")
    config = Config(modelName, relsFolder)
    print ("read config")

    if (config.testFlag):
        test_lib.init()
        config.relation = test_lib.getRelationTotal()
        config.entity = test_lib.getEntityTotal()
        config.batch = test_lib.getEntityTotal()
        config.batch_size = config.batch
    else:
        print ("lib init?")
        lib.init()
        print ("lib init")
        config.relation = lib.getRelationTotal()
        print ("relation...")
        config.entity = lib.getEntityTotal()
        config.batch_size = lib.getTripleTotal() // config.nbatches

    with tf.Graph().as_default():
        sess = tf.Session()
        with sess.as_default():
            print ("in session")
            initializer = tf.contrib.layers.xavier_initializer(uniform=False)
            with tf.variable_scope("model", reuse=None, initializer=initializer):
                trainModel = TransEModel(config=config)

            saver = tf.train.Saver()
            sess.run(tf.initialize_all_variables())
            if (config.loadFromData):
                saver.restore(sess, './' + config.modelName)
                if 1==1:
                    xx = trainModel.rel_embeddings.eval()
                    print (len(xx), type(xx))
                    return xx


if __name__ == "__main__":
    tf.app.run()
