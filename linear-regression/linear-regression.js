const tf = require('@tensorflow/tfjs');
const _ = require('lodash');

class LinearRegression {
    constructor(features, labels, options){
        this.features = this.processFeatures(features);
        this.labels = tf.tensor(labels);
        this.options = Object.assign({learningRate: .1, iterations: 1000}, options)
        this.weights = tf.zeros([this.features.shape[1],1]);
        this.mseHistory = [];

    }
    train() {
        const batchQuantity = Math.floor(this.features.shape[0] / this.options.batchSize)
        for(let i = 0; i < this.options.iterations; i++){            
            for(let j = 0; j < batchQuantity; j++){
                const { batchSize } = this.options;
                const startIndex = j * batchSize;
                const featureSlice = this.features.slice([startIndex, 0], [batchSize, -1])
                const labelSlice = this.labels.slice([startIndex, 0], [batchSize, -1])
                this.gradientDescent(featureSlice, labelSlice);
            }
            this.recordMSE();
            this.updateLearningRate();
        }
    }
    gradientDescent(features, labels) {
        // currentGuess is a matrix multiplication of [[m], [b]] which is equivalent to doing guess/y = mx + b for every feature
        const currentGuesses = features.matMul(this.weights);
        const differences = currentGuesses.sub(labels);

        const slopes = features
            // transpose the feature matrix which is originally row by column tensor/matrix with ones concatenated. Now it's a col by row matrix, where each column is a feature and a value of one, and there are only two rows, one for the features and one for the 1's
            .transpose()
            // multiply the features by the differences btwn (mx + b) - labels
            .matMul(differences)
            // divide by n, or the total number of entries
            .div(features.shape[0])
        this.weights = this.weights.sub(slopes.mul(this.options.learningRate));
    }
//     gradientDescent() {
//         // guessing the MPG (label) using every horse power values (feature)
//         // this.features is an array of arrays
//         const currentGuessesForMPG = this.features.map(row => {
//             //   MPG/y  =   slope/m * current val/x + y intercept/b
//             return this.m * row[0] + this.b;
//         })
//         // take the sum of all MSE with respect to B and subtract current guess, then mult by 2 and divide by N
//         const bSlope = _.sum(currentGuessesForMPG.map( (guess, i) => {
//             return guess - this.labels[i][0]
//         })) * 2 / this.features.length;
//             // divide by n 
//         const mSlope = _.sum(currentGuessesForMPG.map( (guess, i) => {
//             return -1 * this.features[i][0] * (this.labels[i][0] - guess);
//         })) * 2 / this.features.length;

//         this.b = this.b - bSlope * this.options.learningRate;
//         this.m = this.m - mSlope * this.options.learningRate; 
//     }
    test(testFeatures, testLabels) {
        testFeatures = this.processFeatures(testFeatures);
        testLabels = tf.tensor(testLabels);
        const predictions = testFeatures.matMul(this.weights);
        // Equation is Sum of Squares of Residuals / SSres
        const res = testLabels.sub(predictions)
            // raise to the second power, element-wise operation
            .pow(2)
            .sum()
            .get()
        // .sub() is element wise, subtract the mean of all the labels
        const tot = testLabels.sub(testLabels.mean())
            .pow(2)
            .sum()
            .get()
        return 1 - res / tot
    }
    processFeatures(features){
        features = tf.tensor(features);
        if (this.mean && this.variance) {
            features = features.sub(this.mean).div(this.variance.pow(0.5));
        } else {
            features = this.standardize(features);
        }
        // creates a tensor of shape [this.features.shape[0]/rows, 1 col] and concatenates the result to the features tensor along the horizontal/y axis
        features = tf.ones([features.shape[0], 1]).concat(features, 1);
        return features;
    }
    predict(observations) {
        return this.processFeatures(observations).matMul(this.weights);
    }
    standardize(features) {
        const { mean, variance } = tf.moments(features, 0);
        this.mean = mean;
        this.variance = variance;

        return features.sub(mean).div(variance.pow(0.5));
    }
    recordMSE(){
        const mse = this.features
            .matMul(this.weights)
            .sub(this.labels)
            .pow(2)
            .sum()
            .div(this.features.shape[0])
            .get();
        
        this.mseHistory.push(mse);
    }
    updateLearningRate() {
        const mseLength = this.mseHistory.length;
        if(mseLength < 2) return;
        
        const lastValue = this.mseHistory[mseLength - 1];
        const secondToLast = this.mseHistory[mseLength - 2];
        // if the MSE is increasing, divide by 2, else increase by 5%
        if(lastValue > secondToLast){
            this.options.learningRate /= 2;
        } else {
            this.options.learningRate *= 1.05;
        }
    }
}

module.exports = LinearRegression;