const tf = require('@tensorflow/tfjs');
const _ = require('lodash');

class LogisticRegression {
    constructor(features, labels, options){
        this.features = this.processFeatures(features);
        this.labels = tf.tensor(labels);
        this.options = Object.assign({learningRate: .1, iterations: 1000, decisionBoundary: 0.5}, options)
        this.weights = tf.zeros([this.features.shape[1],this.labels.shape[1]]);
        // crossEntropy history
        this.costHistory = [];

    }
    train() {
        const batchQuantity = Math.floor(this.features.shape[0] / this.options.batchSize)
        for(let i = 0; i < this.options.iterations; i++){            
            for(let j = 0; j < batchQuantity; j++){
                const { batchSize } = this.options;
                const startIndex = j * batchSize;
                this.weights = tf.tidy(()=> {
                    const featureSlice = this.features.slice([startIndex, 0], [batchSize, -1])
                    const labelSlice = this.labels.slice([startIndex, 0], [batchSize, -1])
                    return this.gradientDescent(featureSlice, labelSlice);
                })
                
            }
            this.recordCost();
            this.updateLearningRate();
        }
    }
    gradientDescent(features, labels) {
        const currentGuesses = features.matMul(this.weights).softmax();
        const differences = currentGuesses.sub(labels);

        const slopes = features
            .transpose()
            .matMul(differences)
            .div(features.shape[0])
        return this.weights.sub(slopes.mul(this.options.learningRate));
    }
    test(testFeatures, testLabels) {
        const predictions = this.predict(testFeatures);
        testLabels = tf.tensor(testLabels).argMax(1);

        const incorrect = predictions.notEqual(testLabels).sum().get();
        let numOfPredictions =  predictions.shape[0]
        return (numOfPredictions - incorrect) / numOfPredictions;
    }
    processFeatures(features){
        features = tf.tensor(features);
        if (this.mean && this.variance) {
            features = features.sub(this.mean).div(this.variance.pow(0.5));
        } else {
            features = this.standardize(features);
        }
        features = tf.ones([features.shape[0], 1]).concat(features, 1);
        return features;
    }
    predict(observations) {
        return this.processFeatures(observations)
                .matMul(this.weights)
                .softmax()
                .argMax(1)
    }
    standardize(features) {
        const { mean, variance } = tf.moments(features, 0);
        // cast the variance into bool so that the zeros and be flipped to ones, then changed back to floats, This avoids tensorflows issue with dividing by zero and eventualy going up to infinity
        const filler = variance.cast('bool').logicalNot().cast('float32');
        this.mean = mean;
        this.variance = variance.add(filler);

        return features.sub(mean).div(this.variance.pow(0.5));
    }
    recordCost(){
        const cost = tf.tidy(()=> {
            const guesses = this.features.matMul(this.weights).softmax();
            const termOne = this.labels.transpose().matMul(guesses.log());
            const termTwo = this.labels
                .mul(-1)
                .add(1)
                .transpose()
                .matMul(
                    guesses
                        .mul(-1)
                        .add(1)
                        .log()
                );
            return termOne.add(termTwo)
                    .div(this.features.shape[0])
                    .mul(-1)
                    .get(0,0);
        })
        
        this.costHistory.unshift(cost);
    }
    updateLearningRate() {
        const costLength = this.costHistory.length;
        if(costLength < 2) return;
        
        const lastValue = this.costHistory[costLength - 1];
        const secondToLast = this.costHistory[costLength - 2];
        if(lastValue > secondToLast){
            this.options.learningRate /= 2;
        } else {
            this.options.learningRate *= 1.05;
        }
    }
}

module.exports = LogisticRegression;