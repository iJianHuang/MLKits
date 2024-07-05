require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const LogisticRegression = require('./logistic-regression');
const plot = require('node-remote-plot');
const _ = require('lodash');
const mnist = require('mnist-data');
// --inspect-brk
// debugger
// --max-old-space-size=4096

function loadData() {
    const mnistData = mnist.training(0, 60000); // 60000

    const features = mnistData.images.values.map(image => _.flatMap(image));
    const encodedLabels = mnistData.labels.values.map(label => {
        const row = new Array(10).fill(0);
        row[label] = 1;
        return row;
    });

    return { features, labels: encodedLabels };
}

const { features, labels } = loadData();

//console.log(mnistData);
//console.log(mnistData.labels.values);
//console.log(features);
//console.log(encodedLabels);

const regression = new LogisticRegression(features, labels, {
    learningRate: 1,
    iterations: 40,
    batchSize: 500
});


regression.train();
//debugger

const testMnistData = mnist.testing(0, 10000);
const testFeatures = testMnistData.images.values.map(image => _.flatMap(image));
const testEncodedLabels = testMnistData.labels.values.map(label => {
    const row = new Array(10).fill(0);
    row[label] = 1;
    return row;
});

//console.log(testMnistData);
const accuracy = regression.test(testFeatures, testEncodedLabels);
console.log('Accuracy: ', accuracy);

plot({
    x: regression.costHistory.reverse()
});
console.log(regression.costHistory.reverse());

