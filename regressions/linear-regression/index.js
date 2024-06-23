require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCSV = require('../load-csv');
const LinearRegression = require('./linear-regression');
const plot = require('node-remote-plot');

let { features, labels, testFeatures, testLabels } = loadCSV('../data/cars.csv', {
    shuffle: true,
    splitTest: 50,
    dataColumns: ['horsepower', 'weight', 'displacement'],
    labelColumns: ['mpg']
});

const regression = new LinearRegression(features, labels, {
    learningRate: .1,
    iterations: 3,
    batchSize: 10
});

regression.train();
r2 = regression.test(testFeatures, testLabels);  

plot({
    x: regression.mseHistory,
    path: "C://Users//Jian//Documents//GitHub//MLKits//regressions"
});
console.log('r2: ', r2);

const observations = [
    [120, 2, 380],
    [135, 3.2, 420]
];
const predictions = regression.predict(observations);
predictions.print();
