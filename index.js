require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCSV = require('./load-csv');
const LinearRegression = require('./linear-regression');
const plot = require('node-remote-plot');

let {features, labels, testFeatures, testLabels} = loadCSV('./cars.csv', {
    shuffle: true,
    splitTest: 50,
    dataColumns: ['horsepower', 'weight', 'displacement'],
    labelColumns: ['mpg']
})

const regression = new LinearRegression(features, labels, {
    learningRate: 0.1,
    iterations: 100
}, testFeatures, testLabels)

regression.train();

// r^2 or Coefficient of Determination
const r2 = regression.test(testFeatures, testLabels);

plot({
    x: regression.mseHistory,
    xLabel: 'Iteration #',
    yLabel: 'Mean Squared Error'
})