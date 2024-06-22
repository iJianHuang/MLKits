const tf = require('@tensorflow/tfjs');
const _ = require('lodash');

class LinearRegression {
    constructor(features, labels, options) {
        this.features = this.processFeatures(features);
        this.labels = tf.tensor(labels);
       
        this.options = Object.assign(
            { learningRate: 0.1, iterations: 1000 }, 
            options
        ); 

        this.weights = tf.zeros([this.features.shape[1], 1]);
    }

    gradientDescent() {
        // transposed-features * ((features-w-ones * weights) - labels)
        const currentGuesses = this.features.matMul(this.weights);
        const differences = currentGuesses.sub(this.labels);
        const slopes = this.features
            .transpose()
            .matMul(differences)
            .div(this.features.shape[0]);

        this.weights = this.weights
            .sub(slopes.mul(this.options.learningRate));
               
    }    
    
    train() {
        for (let i = 0; i < this.options.iterations; i++ ){
            this.gradientDescent();
        }
    }  
    
    test(testFeatures, testLabels) {
        testFeatures = this.processFeatures(testFeatures);
        testLabels = tf.tensor(testLabels);        

        const predictions = testFeatures.matMul(this.weights);
        
        const ssRes = testLabels
            .sub(predictions)
            .pow(2)
            .sum()
            .get();

        const ssTot = testLabels
            .sub(testLabels.mean())
            .pow(2)
            .sum()
            .get();   

        return 1 - ssRes / ssTot;    
    }

    processFeatures(features) {
        features = tf.tensor(features);
                
        if (this.mean & this.variance) {
            features = features
                .sub(this.mean)
                .div(this.variance.pow(.5));
        } else {
            features = this.standardize(features);
        }  

        features = tf
            .ones([features.shape[0], 1]) // 2D tensor with ones
            .concat(features, 1);
        return features;
    }

    standardize(features) {
        const { mean, variance } = tf.moments(features, 0);
        this.mean = mean;
        this.variance = variance;
        return features
            .sub(mean)
            .div(variance.pow(.5));
    }
}

module.exports = LinearRegression;