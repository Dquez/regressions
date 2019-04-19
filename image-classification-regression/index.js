require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const _ = require('lodash');
const LogisticRegression = require('./logistic-regression');
const mnist = require('mnist-data');

// grab the first ten numbers from training data
const mnistData = mnist.training(0,5000);

// map over numbers to create 1 condensed array for each number with corresponding grayscale values.
const features = mnistData.images.values.map(image => _.flatMap(image));
// encode labels to have a 1 for the corresponding label
const encodedLabels = mnistData.labels.values.map(label=> {
    const row = new Array(10).fill(0);
    row[label] = 1;
    return row;
})

const regression = new LogisticRegression(features, encodedLabels, {
    learningRate: 1,
    iterations: 20,
    batchSize: 100
})

regression.train();

const testMnistData = mnist.testing(0,100);
const testFeatures = testMnistData.images.values.map(image => _.flatMap(image));
const testEncodedLabels = testMnistData.labels.values.map(label=> {
    const row = new Array(10).fill(0);
    row[label] = 1;
    return row;
})

const accuracy = regression.test(testFeatures, testEncodedLabels);
console.log('Accuracy is ', accuracy);
