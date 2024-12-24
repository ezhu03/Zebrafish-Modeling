import numpy as np

class HiddenMarkovModel:
    def __init__(self, num_states, num_observations):
        self.num_states = num_states
        self.num_observations = num_observations
        self.transition_matrix = np.zeros((num_states, num_states))
        self.emission_matrix = np.zeros((num_states, num_observations))
        self.initial_probabilities = np.zeros(num_states)
        
    def fit(self, observations, num_iterations=100, epsilon=1e-8):
        # Initialize model parameters with pseudocounts
        self.transition_matrix = np.ones((self.num_states, self.num_states)) * epsilon
        self.emission_matrix = np.ones((self.num_states, self.num_observations)) * epsilon
        self.initial_probabilities = np.ones(self.num_states) * epsilon
        
        # Training the model using the Baum-Welch algorithm (Expectation-Maximization)
        for _ in range(num_iterations):
            alpha, beta, gamma, xi = self._expectation_step(observations)
            self._maximization_step(observations, alpha, beta, gamma, xi, epsilon)
    
    def _forward_pass(self, observations, epsilon):
        T = len(observations)
        alpha = np.zeros((T, self.num_states))
        alpha[0] = np.log(self.initial_probabilities + epsilon) + np.log(self.emission_matrix[:, observations[0]] + epsilon)
        
        for t in range(1, T):
            for j in range(self.num_states):
                alpha[t, j] = np.logaddexp.reduce(alpha[t - 1] + np.log(self.transition_matrix[:, j] + epsilon)) + \
                              np.log(self.emission_matrix[j, observations[t]] + epsilon)
                
        return alpha
    
    def _backward_pass(self, observations, epsilon):
        T = len(observations)
        beta = np.zeros((T, self.num_states))
        beta[-1] = 0.0
        
        for t in range(T - 2, -1, -1):
            for i in range(self.num_states):
                beta[t, i] = np.logaddexp.reduce(np.log(self.transition_matrix[i, :] + epsilon) +
                                                 np.log(self.emission_matrix[:, observations[t + 1]] + epsilon) +
                                                 beta[t + 1])
        
        return beta
    
    def _expectation_step(self, observations):
        epsilon = 1e-8
        alpha = self._forward_pass(observations, epsilon)
        beta = self._backward_pass(observations, epsilon)
        T = len(observations)
        
        gamma = alpha + beta
        gamma -= np.max(gamma, axis=1, keepdims=True)  # Normalize to avoid numerical instability
        gamma = np.exp(gamma)
        gamma /= np.sum(gamma, axis=1, keepdims=True)
        
        xi = np.zeros((T - 1, self.num_states, self.num_states))
        for t in range(T - 1):
            for i in range(self.num_states):
                for j in range(self.num_states):
                    xi[t, i, j] = alpha[t, i] + np.log(self.transition_matrix[i, j] + epsilon) + \
                                 np.log(self.emission_matrix[j, observations[t + 1]] + epsilon) + beta[t + 1, j]
            xi[t] = xi[t] - np.max(xi[t])  # Normalize to avoid numerical instability
            xi[t] = np.exp(xi[t])
            xi[t] /= xi[t].sum()
        
        return alpha, beta, gamma, xi
    
    def _maximization_step(self, observations, alpha, beta, gamma, xi, epsilon):
        T = len(observations)
        
        # Update transition matrix
        for i in range(self.num_states):
            for j in range(self.num_states):
                self.transition_matrix[i, j] = np.sum(xi[:, i, j]) / np.sum(gamma[:, i])
        
        # Update emission matrix
        for j in range(self.num_states):
            for k in range(self.num_observations):
                mask = observations == k
                self.emission_matrix[j, k] = np.sum(gamma[:, j] * mask) / np.sum(gamma[:, j])
        
        # Update initial probabilities
        self.initial_probabilities = gamma[0]
        
# Rest of the code remains the same...

        
# Rest of the code remains the same...
    def predict(self, observations):
        # Use the Viterbi algorithm to find the most likely state sequence given the observations
        T = len(observations)
        delta = np.zeros((T, self.num_states))
        psi = np.zeros((T, self.num_states), dtype=int)
        delta[0] = self.initial_probabilities * self.emission_matrix[:, observations[0]]
        
        for t in range(1, T):
            for j in range(self.num_states):
                temp = delta[t - 1] * self.transition_matrix[:, j] * self.emission_matrix[j, observations[t]]
                psi[t, j] = np.argmax(temp)
                delta[t, j] = np.max(temp)
        
        # Backtrack to find the most likely sequence
        state_sequence = [np.argmax(delta[-1])]
        for t in range(T - 1, 0, -1):
            state_sequence.append(psi[t, state_sequence[-1]])
        state_sequence.reverse()
        
        return state_sequence

# Example usage:
# Define the number of states and observations
num_states = 2
num_observations = 3

# Create the HMM model
hmm = HiddenMarkovModel(num_states, num_observations)

# Generate some example observations (assuming discrete observations)
observations = [0, 1, 2, 1, 0, 2, 2, 1]

# Fit the model to the observations
hmm.fit(observations)

# Predict the most likely state sequence for the observations
predicted_states = hmm.predict(observations)
print("Predicted States:", predicted_states)