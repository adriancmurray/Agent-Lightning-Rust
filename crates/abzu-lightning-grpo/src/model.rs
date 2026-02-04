use candle_core::{Result, Tensor, Module};
use candle_nn::{Linear, VarBuilder, linear};

/// Policy-only model for GRPO (No Critic)
pub struct Actor {
    layers: Vec<Linear>,
    head: Linear,
}

impl Actor {
    pub fn new(vb: VarBuilder, input_dim: usize, action_dim: usize) -> Result<Self> {
        let hidden_size = 64;

        // Actor Network: Linear(input, 64) -> Tanh -> Linear(64, 64) -> Tanh -> Linear(64, action)
        let l1 = linear(input_dim, hidden_size, vb.pp("actor.0"))?;
        let l2 = linear(hidden_size, hidden_size, vb.pp("actor.1"))?;
        let head = linear(hidden_size, action_dim, vb.pp("actor.head"))?;

        Ok(Self {
            layers: vec![l1, l2],
            head,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = x.clone();
        for layer in &self.layers {
            h = layer.forward(&h)?;
            h = h.tanh()?;
        }
        self.head.forward(&h)
    }
}
