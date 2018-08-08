# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 09:31:42 2018

@author: Philip.Osterkamp
"""

import mnist_loader
import network 

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net =network.Network([784,30,10])
net.SGD(training_data, 30, 100, 10.0, test_data=test_data)

