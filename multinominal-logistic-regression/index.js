require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const _ = require('lodash');
const loadCSV = require('../loadCSV/load-csv');
const LogisticRegression = require('./logistic-regression');
const plot = require('node-remote-plot');

let {features, labels, testFeatures, testLabels} = loadCSV('../data/cars.csv', {
    shuffle: true,
    splitTest: 50,
    dataColumns: ['horsepower', 'displacement', 'weight'],
    labelColumns: ['mpg'],
    converters: {
        mpg: value => {
            const mpg = parseFloat(value);
            // The mpg continuous values are broken into three groups, low medium and high fuel efficiency, marked by an array
            return  mpg < 15 ? [1,0,0] :
                    mpg < 30 ? [0,1,0] :
                    [0,0,1]
        }
    }
})

const regression = new LogisticRegression(features, _.flatMap(labels), {
    learningRate: 0.5,
    iterations: 100,
    batchSize: 10
})

regression.train();
console.log(regression.test(testFeatures, _.flatMap(testLabels)));