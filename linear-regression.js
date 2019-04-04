const tf = require('@tensorflow/tfjs');
const _ = require('lodash');

class LinearRegression {
    constructor(features, labels, options){
        this.features = this.processFeatures(features);
        this.labels = tf.tensor(labels);
        this.options = Object.assign({learningRate: .1, iterations: 1000}, options)
        this.weights = tf.zeros([2,1]);
        
    }
    train() {
        for(let i = 0; i < this.options.iterations; i++){
            this.gradientDescent();
        }
    }
    gradientDescent() {
        // currentGuess is a matrix multiplication of [[m], [b]] which is equivalent to doing guess/y = mx + b for every feature
        const currentGuesses = this.features.matMul(this.weights);
        const differences = currentGuesses.sub(this.labels);

        const slopes = this.features
            // transpose the feature matrix which is originally row by column tensor/matrix with ones concatenated. Now it's a col by row matrix, where each column is a feature and a value of one, and there are only two rows, one for the features and one for the 1's
            .transpose()
            // multiply the features by the differences btwn (mx + b) - labels
            .matMul(differences)
            // divide by n, or the total number of entries
            .div(this.features.shape[0])
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
        // creates a tensor of shape [this.features.shape[0]/rows, 1 col] and concatenates the result to the features tensor along the horizontal/y axis
        features = tf.ones([features.shape[0], 1]).concat(features, 1);

        return features;
    }
}

module.exports = LinearRegression;