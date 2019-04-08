require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCSV = require('../load-csv');
const LogisticRegression = require('./logistic-regression');

let {features, labels, testFeatures, testLabels} = loadCSV('../data/cars.csv', {
    shuffle: true,
    splitTest: 50,
    dataColumns: ['horsepower', 'displacement', 'weight'],
    labelColumns: ['passedemissions'],
    converters: {
        passedemissions: value => value === 'TRUE' ? 1 : 0
    }
})

const regression = new LogisticRegression(features, labels, {
    learningRate: 0.5,
    iterations: 100,
    batchSize: 50
})

regression.train();
regression.predict([
    [88, 97, 1.065]
]).print()