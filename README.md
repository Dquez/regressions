# Regressions

Regressions is a multivariate regression algorithm that predicts the mpg of a car based on certain values, namely horsepower, weight, and displacement. It's broken into three separate types of regression, denoted by their folder respective folders: linear, logisitic and multinominal regression.

## Built With

* [tensorflow.js](https://www.tensorflow.org/js) - TensorFlow.js is a library for developing and training ML models in JavaScript, and deploying in the browser or on Node.js


## Inspiration
This is a side application I wanted to build to better understand basic ML algorithms. With the fundamentals, I'm able to explore and tinker with the tensorflow library.

## How I built it
This app was bootstrapped with the [Machine Learning with Javascript](https://www.udemy.com/machine-learning-with-javascript/learn/lecture/12279722#overview) course on Udemy. The CSV loader and data was already implemented but started the process by creating a LinearRegression class and adding methods to find a reliable MPG prediction based on one feature, horsepower. At first, I implemented the gradient descent algorithm with for loops and array manipulation, but that method wasn't scalable so I refactored the class to use Tensorflow instead. This way, despite multiple features, I was still able to come up with an accurate MPG prediction. My process:
 * Determine a reasonable test and training size, along with a low enough learning rate to avoid bounce between outlier Mean Squared Error calculations. 
 * Develop a rudimentary algorithm to determine gradient descent using arrays and for loops.
 * Refactor using tensors and matrix manipulation.
 * Concatenate a tensor of 1's to the features tensor to ensure the eventual mx + b or b + mx formula works as expected.
 * Take weights (*m and b*) from instance variables to tensor, to allow for matrix manipulation.
 * Update weights based on the product of multiplying the transpose matrix of features to the difference between the current label and the product of features * weights. (*this step occurs during the gradient descent method, which updates our predictions weights to find the optimal gradient with least slope*)
 * Standardize the features using standard deviation
 * Add a predict function to determine if values we input are in line with what the expected MPG is.
 * Record MSE (*Mean Squared Error*) to update the learning rate based on changes to the MSE.
 * Finally, update MSE using Batch Gradient Descent 


## Challenges I ran into
* Develop a meaningful prediction that was at least better than taking the average MPG for all cars.
* Understanding tensorflow and how to manipulate matrices of varying shapes together to get the mx + b formula working.
* Determining which form of normalization of values after adding multiple features into the equation.

## Accomplishments that I'm proud of
In this current iteration, I'm using multiple methods of the tensor flow library yet the functions aren't all obfuscated. With the exception statistical theory, I can confidently follow the logic behind the matrix manipulation. Most importantly, I have a working solution that accurately predicts an MPG value based on certain features of the data set.

## Author

- [Dariell Vasquez](https://github.com/Dquez)