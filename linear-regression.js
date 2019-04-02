const tf = require('@tensorflow/tfjs-node');
const _ = require('lodash');

class LinearRegression {
    constructor(features, labels, options){
        this.features = features;
        this.labels = labels;
        this.options = Object.assign({learningRate: .1, iterations: 1000}, options)
        this.m = 0;
        this.b = 0;
    }
    train() {
        for(let i = 0; i < this.options.iterations; i++){
            this.gradientDescent();
        }
    }
    gradientDescent() {
        // guessing the MPG (label) using every horse power values (feature)
        // this.features is an array of arrays
        const currentGuessesForMPG = this.features.map(row => {
            //   MPG/y  =   slope/m * current val/x + y intercept/b
            return this.m * row[0] + b;
        })
        // take the sum of all MSE with respect to B and subtract current guess, then mult by 2 and divide by N
        const bSlope = _.sum(currentGuessesForMPG.map( (guess, i) => {
            return guess - this.labels[i][0]
        })) * 2 / this.features.length;
            // divide by n 
        const mSlope = _.sum(currentGuessesForMPG.map( (guess, i) => {
            return -1 * this.features[i][0] * (this.labels[i][0] - guess);
        })) * 2 / this.features.length;

        this.b = this.b - bSlope * this.options.learningRate;
        this.m = this.m - mSlope * this.options.learningRate; 
    }
}

module.export = LinearRegression;