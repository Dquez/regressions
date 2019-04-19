require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const _ = require('lodash');
const loadCSV = require('../load-csv');
const LogisticRegression = require('./logistic-regression');
const plot = require('node-remote-plot');
const mnist = require('mnist-data');

// grab the first ten numbers from training data
const mnistData = mnist.training(0,10);

// map over numbers to create 1 condensed array for each number with corresponding grayscale values.
const features = mnistData.images.values.map(image => _.flatMap(image));
// encode labels to have a 1 for the corresponding label
const encodedLabels = mnistData.labels.values.map(label=> {
    const row = new Array(10).fill(0);
    row[label] = 1;
    return row;
})

console.log(encodedLabels);