import random
import math

class PatientRecord:
    def __init__(self, age, glucose, comorbidity, systolic_bp, risk_outcome):
        self.features = [age, glucose, comorbidity, systolic_bp]
        self.outcome = risk_outcome

class ClinicalPipelineSimulation:
    def __init__(self):
        self.k_folds = 5
        self.learning_rate = 0.1
        self.iterations = 1000

    def generate_data(self, count):
        """Generates synthetic deep-phenotype data."""
        data = []
        for _ in range(count):
            age = random.randint(30, 79)
            glucose = random.uniform(90, 140)
            comorbidity = random.uniform(0.1, 1.0)
            bp = random.uniform(110, 150)
            
            # Underlying data-generating distribution
            risk = 1 if (age * 0.02 + glucose * 0.01 + comorbidity * 2.0 + bp * 0.01 + random.uniform(0, 0.5)) > 3.5 else 0
            data.append(PatientRecord(age, glucose, comorbidity, bp, risk))
        return data

    def z_score_standardize(self, train_data, test_data):
        """Standardizes features to have mean=0 and variance=1 based on training data."""
        num_features = len(train_data[0].features)
        means = [0.0] * num_features
        stdevs = [0.0] * num_features
        n = len(train_data)

        # Calculate Means
        for p in train_data:
            for i in range(num_features):
                means[i] += p.features[i]
        means = [m / n for m in means]

        # Calculate Standard Deviations
        for p in train_data:
            for i in range(num_features):
                stdevs[i] += (p.features[i] - means[i]) ** 2
        stdevs = [math.sqrt(s / n) if s > 0 else 1.0 for s in stdevs]

        # Apply to both train and test (preventing data leakage)
        def scale(dataset):
            scaled_X = []
            Y = []
            for p in dataset:
                scaled_x = [(p.features[i] - means[i]) / stdevs[i] for i in range(num_features)]
                scaled_X.append(scaled_x)
                Y.append(p.outcome)
            return scaled_X, Y

        train_X, train_Y = scale(train_data)
        test_X, test_Y = scale(test_data)
        return train_X, train_Y, test_X, test_Y

class LogisticRegressionFromScratch:
    def __init__(self, num_features):
        self.weights = [0.0] * num_features
        self.bias = 0.0

    def _sigmoid(self, z):
        # Math domain error protection for overflow
        z = max(min(z, 250), -250) 
        return 1.0 / (1.0 + math.exp(-z))

    def predict_prob(self, features):
        z = self.bias + sum(w * x for w, x in zip(self.weights, features))
        return self._sigmoid(z)

    def train(self, X, Y, lr, epochs):
        m = len(X)
        num_features = len(self.weights)
        
        for _ in range(epochs):
            bias_grad = 0.0
            weight_grads = [0.0] * num_features
            
            for i in range(m):
                pred = self.predict_prob(X[i])
                error = pred - Y[i]
                
                bias_grad += error
                for j in range(num_features):
                    weight_grads[j] += error * X[i][j]
                    
            # Gradient descent update step
            self.bias -= lr * (bias_grad / m)
            for j in range(num_features):
                self.weights[j] -= lr * (weight_grads[j] / m)

def main():
    print("=== Python ML Pipeline: Clinical Risk Prediction ===")
    pipeline = ClinicalPipelineSimulation()
    dataset = pipeline.generate_data(500)
    
    # Shuffle for K-Fold
    random.shuffle(dataset)
    fold_size = len(dataset) // pipeline.k_folds
    
    total_acc = 0.0
    
    print(f"Running {pipeline.k_folds}-Fold Cross Validation with Batch Gradient Descent...\n")
    
    for k in range(pipeline.k_folds):
        # Split Data
        test_start = k * fold_size
        test_end = test_start + fold_size
        test_data = dataset[test_start:test_end]
        train_data = dataset[:test_start] + dataset[test_end:]
        
        # Standardize strictly on training distribution
        train_X, train_Y, test_X, test_Y = pipeline.z_score_standardize(train_data, test_data)
        
        # Train Model
        model = LogisticRegressionFromScratch(num_features=4)
        model.train(train_X, train_Y, pipeline.learning_rate, pipeline.iterations)
        
        # Evaluate
        correct = 0
        for x, y_true in zip(test_X, test_Y):
            y_pred = 1 if model.predict_prob(x) >= 0.5 else 0
            if y_pred == y_true:
                correct += 1
                
        acc = correct / len(test_data)
        total_acc += acc
        print(f"Fold {k+1} Accuracy: {acc * 100:.2f}%")
        
    print("-" * 50)
    print(f"Average Model Accuracy: {(total_acc / pipeline.k_folds) * 100:.2f}%")
    print("Pipeline Complete.")

if __name__ == "__main__":
    main()
