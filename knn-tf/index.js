require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCSV = require('./load-csv');

/*
node --inspect-brk index.js
about:Inspect
*/

function knn(features, labels, predictionPoint, k) {
    const {mean, variance} = tf.moments(features, 0); // along 0 (column axis)
    // (value - mean) / sq root of variance
    const scaledPrediction = predictionPoint.sub(mean).div(variance.pow(.5));

    return 	features
        .sub(mean)
        .div(variance.pow(.5))
        .sub(scaledPrediction)
        .pow(2)
        .sum(1)  // 1 axis, and reduce dimension, become 1D
        .pow(.5) // distance of M dimensions
        .expandDims(1) // expand Dims into 2D, shape [n,1]
        .concat(labels, 1) 
        .unstack() // convert into tensor array
        .sort((a,b) => a.get(0) > b.get(0)? 1:-1) // get value from tensor
        .slice(0, k) 
        .reduce((acc,pair) => acc + pair.get(1), 0)
		    / k; 
}

let {features, labels, testFeatures, testLabels} = loadCSV('kc_house_data.csv', {
    shuffle: true,
    splitTest: 10,
    dataColumns: ['lat','long','sqft_lot','sqft_living'],
    labelColumns: ['price']
});

features = tf.tensor(features);
labels = tf.tensor(labels);

testFeatures.forEach((testPoint, i) => {
    const result = knn(features, labels,tf.tensor(testPoint),10);
    const err = (testLabels[i][0] - result ) / testLabels[i][0];

    console.log('Error', err * 100);    
});
