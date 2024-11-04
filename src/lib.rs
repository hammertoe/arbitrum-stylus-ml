#![cfg_attr(not(feature = "export-abi"), no_main)]

extern crate alloc;
use alloc::vec::Vec;
use alloc::vec;
use stylus_sdk::{alloy_primitives::U256, prelude::*};


const SCALE_FACTOR: i32 = 10000; // Fixed-point scaling factor

/// Activation functions
fn relu(x: i32) -> i32 {
    if x > 0 { x } else { 0 }
}

fn sigmoid(x: i32) -> i32 {
    if x < -6 * SCALE_FACTOR {
        0
    } else if x > 6 * SCALE_FACTOR {
        SCALE_FACTOR
    } else {
        SCALE_FACTOR / 2 + (x / 12)
    }
}

/// Multi-Layer Perceptron with integer weights
pub struct MLP {
    weights: Vec<Vec<Vec<i32>>>, // [layer][neuron][weight]
    biases: Vec<Vec<i32>>,       // [layer][neuron]
}

impl MLP {
    pub fn new(weights: Vec<Vec<Vec<i32>>>, biases: Vec<Vec<i32>>) -> Self {
        MLP { weights, biases }
    }

    pub fn predict(&self, input: &[i32]) -> i32 {
        let mut layer_input = input.to_vec();

        for (layer_idx, (layer_weights, layer_biases)) in self.weights.iter().zip(self.biases.iter()).enumerate() {
            let mut layer_output = Vec::new();

            for (neuron_idx, neuron_weights) in layer_weights.iter().enumerate() {
                let mut sum = layer_biases[neuron_idx];

                for (&w, &x) in neuron_weights.iter().zip(layer_input.iter()) {
                    sum += w * x / SCALE_FACTOR;
                }

                let activated = if layer_idx == self.weights.len() - 1 {
                    sigmoid(sum)
                } else {
                    relu(sum)
                };

                layer_output.push(activated);
            }

            layer_input = layer_output;
        }

        let output = layer_input[0];

        if output >= SCALE_FACTOR / 2 {
            1
        } else {
            0
        }
    }
}

/// Function to load the quantized model parameters
fn load_model() -> MLP {
    let weights = vec![
        // Layer 1 weights
        vec![
            vec![1000, -2000, 1500],
            vec![-1500, 2500, -1000],
        ],
        // Layer 2 weights (output layer)
        vec![
            vec![2000, -3000],
        ],
    ];

    let biases = vec![
        // Layer 1 biases
        vec![5000, -5000],
        // Layer 2 biases (output layer)
        vec![1000],
    ];

    MLP::new(weights, biases)
}

#[storage]

#[entrypoint]

pub struct Model {
}

#[public]
impl Model {
    /// Function to predict the output of the model
    pub fn predict(&self, input: Vec<i32>) -> U256 {
        let model = load_model();
        let res = model.predict(&input);
        U256::from(res)
    }
    pub fn predict2(&self) -> U256 {
        U256::from(1)
    }
}
