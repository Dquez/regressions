require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const _ = require('lodash');
const loadCSV = require('../load-csv');
const LogisticRegression = require('./logistic-regression');
const plot = require('node-remote-plot');
const mnist = require('mnist-data');
