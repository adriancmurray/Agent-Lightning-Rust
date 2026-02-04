use candle_core::{Result, Tensor, Module};
use candle_nn::{Linear, VarBuilder, linear};

pub struct ActorCritic {
    actor_layers: Vec<Linear>,
    actor_head: Linear,
    critic_layers: Vec<Linear>,
    critic_head: Linear,
}

impl ActorCritic {
    pub fn new(vb: VarBuilder, input_dim: usize, action_dim: usize) -> Result<Self> {
        let hidden_size = 64;

        // Actor Network: Linear(input, 64) -> Tanh -> Linear(64, 64) -> Tanh -> Linear(64, action)
        let actor1 = linear(input_dim, hidden_size, vb.pp("actor.0"))?;
        let actor2 = linear(hidden_size, hidden_size, vb.pp("actor.1"))?;
        let actor_head = linear(hidden_size, action_dim, vb.pp("actor.head"))?;

        // Critic Network: Linear(input, 64) -> Tanh -> Linear(64, 64) -> Tanh -> Linear(64, 1)
        let critic1 = linear(input_dim, hidden_size, vb.pp("critic.0"))?;
        let critic2 = linear(hidden_size, hidden_size, vb.pp("critic.1"))?;
        let critic_head = linear(hidden_size, 1, vb.pp("critic.head"))?;

        Ok(Self {
            actor_layers: vec![actor1, actor2],
            actor_head,
            critic_layers: vec![critic1, critic2],
            critic_head,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        // Actor Forward Pass
        let mut h = x.clone();
        for layer in &self.actor_layers {
            h = layer.forward(&h)?;
            h = h.tanh()?;
        }
        let logits = self.actor_head.forward(&h)?;

        // Critic Forward Pass
        let mut v = x.clone();
        for layer in &self.critic_layers {
            v = layer.forward(&v)?;
            v = v.tanh()?;
        }
        let value = self.critic_head.forward(&v)?;

        Ok((logits, value))
    }
}
