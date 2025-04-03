use std::fs;
use std::path::Path;
use candle_nn::{VarBuilder, Module, Optimizer, AdamW, ParamsAdamW, loss};
use candle_transformers::models::bert::{BertModel, Config, HiddenAct, DTYPE};
use candle_core::{Tensor, Device, Result, DType};
use anyhow::{Error, Ok};
use tokenizers::{PaddingParams, Tokenizer};
use serde_json::Value;

struct Args {
    /// Run on CPU rather than on GPU.
    cpu: bool,
    /// Enable tracing (generates a trace-timestamp.json file).
    tracing: bool,
    /// When set, compute embeddings for this prompt.
    prompt: Option<String>,
    /// The number of times to run the prompt.
    n: usize,
    /// L2 normalization for embeddings.
    normalize_embeddings: bool,
    /// Use tanh based approximation for Gelu instead of erf implementation.
    approximate_gelu: bool,
    /// Training related parameters
    learning_rate: f64,
    epochs: usize,
    batch_size: usize,
    save_checkpoint_path: Option<String>,
    load_checkpoint_path: Option<String>,
}

struct TrainingData {
    token_ids: Tensor,
    token_type_ids: Tensor,
    attention_mask: Tensor,
    labels: Tensor,
}

fn load_training_data(tokenizer: &Tokenizer, device: &Device, batch_size: usize) -> anyhow::Result<TrainingData, Error> {
    // For this example, we'll create a simple training dataset
    // In a real application, you would load your actual training data
    let sentences = [
        "The cat sits outside",
        "A man is playing guitar",
        "I love pasta",
        "The new movie is awesome",
        "The cat plays in the garden",
        "A woman watches TV",
        "The new movie is so great",
        "Do you like pizza?",
    ];
    
    // Creating sample labels (0 and 1 for binary classification)
    // Adjust according to your task (classification, regression, etc.)
    let label_values = [0, 1, 0, 1, 0, 1, 1, 0];
    
    if let Some(pp) = tokenizer.get_padding_mut() {
        pp.strategy = tokenizers::PaddingStrategy::BatchLongest
    } else {
        let pp = PaddingParams {
            strategy: tokenizers::PaddingStrategy::BatchLongest,
            ..Default::default()
        };
        tokenizer.with_padding(Some(pp));
    }
    
    let tokens = tokenizer
        .encode_batch(sentences.to_vec(), true)
        .map_err(Error::msg)?;
    
    let token_ids = tokens
        .iter()
        .map(|tokens| {
            let tokens = tokens.get_ids().to_vec();
            Ok(Tensor::new(tokens.as_slice(), device)?)
        })
        .collect::<Result<Vec<_>>>()?;
    
    let attention_mask = tokens
        .iter()
        .map(|tokens| {
            let tokens = tokens.get_attention_mask().to_vec();
            Ok(Tensor::new(tokens.as_slice(), device)?)
        })
        .collect::<Result<Vec<_>>>()?;

    let token_ids = Tensor::stack(&token_ids, 0)?;
    let attention_mask = Tensor::stack(&attention_mask, 0)?;
    let token_type_ids = token_ids.zeros_like()?;
    
    // Convert labels to tensor
    let labels = Tensor::new(label_values, device)?;
    
    println!("Training data loaded: {} samples", sentences.len());
    println!("Token IDs shape: {:?}", token_ids.shape());
    
    Ok(TrainingData {
        token_ids,
        token_type_ids,
        attention_mask,
        labels,
    })
}

// Classification head for BERT
struct BertClassifier {
    bert: BertModel,
    classifier: candle_nn::Linear,
    n_classes: usize,
}

impl BertClassifier {
    fn new(bert: BertModel, hidden_size: usize, n_classes: usize, device: &Device) -> Result<Self, Error> {
        let classifier = candle_nn::linear(hidden_size, n_classes, device)?;
        Ok(Self { bert, classifier, n_classes })
    }
    
    fn forward(&self, 
               token_ids: &Tensor, 
               token_type_ids: &Tensor, 
               attention_mask: Option<&Tensor>) -> Result<Tensor, Error> {
        // Get the BERT embeddings
        let embeddings = self.bert.forward(token_ids, token_type_ids, attention_mask)?;
        
        // Extract the [CLS] token embedding (first token)
        let cls_embedding = embeddings.get((.., 0, ..))?;
        
        // Pass through the classification layer
        let logits = self.classifier.forward(&cls_embedding)?;
        
        Ok(logits)
    }
    
    fn save(&self, path: &Path) -> Result<(), Error> {
        // Save model weights
        // In a real implementation, you would save all parameters here
        println!("Saving model to {:?}", path);
        // This is a simplified version - you would need to implement actual saving
        Ok(())
    }
}

fn train_model(
    model: &mut BertClassifier,
    data: &TrainingData,
    optimizer: &mut dyn Optimizer,
    epochs: usize,
    batch_size: usize,
    device: &Device,
) -> Result<(), Error> {
    let num_samples = data.token_ids.dim(0)?;
    let num_batches = (num_samples + batch_size - 1) / batch_size;
    
    for epoch in 0..epochs {
        // Track loss for current epoch
        let mut epoch_loss = 0.0;
        
        for batch_idx in 0..num_batches {
            let start_idx = batch_idx * batch_size;
            let end_idx = std::cmp::min(start_idx + batch_size, num_samples);
            
            // Extract mini-batch
            let batch_token_ids = data.token_ids.narrow(0, start_idx, end_idx - start_idx)?;
            let batch_token_type_ids = data.token_type_ids.narrow(0, start_idx, end_idx - start_idx)?;
            let batch_attention_mask = data.attention_mask.narrow(0, start_idx, end_idx - start_idx)?;
            let batch_labels = data.labels.narrow(0, start_idx, end_idx - start_idx)?;
            
            // Forward pass
            let logits = model.forward(&batch_token_ids, &batch_token_type_ids, Some(&batch_attention_mask))?;
            
            // Calculate loss
            let loss = candle_nn::loss::cross_entropy(&logits, &batch_labels)?;
            let loss_val = loss.to_scalar::<f32>()?;
            epoch_loss += loss_val as f64;
            
            // Backward pass and optimize
            optimizer.backward_step(&loss)?;
        }
        
        // Average loss for this epoch
        let avg_loss = epoch_loss / num_batches as f64;
        println!("Epoch {}/{}: Loss = {:.6}", epoch + 1, epochs, avg_loss);
    }
    
    Ok(())
}

fn test() -> Result<(), Box<dyn std::error::Error>> {
    use std::result::Result::Ok;
    let config_str = fs::read_to_string("tokenizer_config.json")?;
    let json_value: Value = match serde_json::from_str(&config_str) {
        Ok(value) => value,
        Err(e) => {
            eprintln!("JSON parse error: {}", e);
            return Err(Box::new(e));
        }
    };
    println!("Parsed JSON: {:?}", json_value);
    Ok(())
}

fn main() -> Result<(), Error> {
    let args = Args {
        cpu: true,
        tracing: true,
        n: 3,
        prompt: None,
        normalize_embeddings: true,
        approximate_gelu: true,
        learning_rate: 5e-5,
        epochs: 3,
        batch_size: 4,
        save_checkpoint_path: Some("bert_checkpoint.bin".to_string()),
        load_checkpoint_path: None,
    };
    
    let device = Device::Cpu;
    
    // Load configuration
    let config_str = std::fs::read_to_string("config.json")?;
    let mut config: Config = serde_json::from_str(&config_str)?;
    
    // Update config for training if needed
    if args.approximate_gelu {
        config.hidden_act = HiddenAct::GeluApproximate;
    }
    
    // Load tokenizer
    let mut tokenizer = Tokenizer::from_file("tokenizer.json").map_err(Error::msg)?;
    
    // Initialize the model
    println!("Loading BERT model...");
    let vb = VarBuilder::from_pth("pytorch_model.bin", DTYPE, &device)?;
    let bert_model = BertModel::load(vb, &config)?;
    
    // Create the classifier (binary classification in this example)
    let n_classes = 2;
    let mut classifier = BertClassifier::new(
        bert_model, 
        config.hidden_size as usize, 
        n_classes, 
        &device
    )?;
    
    // Load training data
    println!("Loading training data...");
    let training_data = load_training_data(&tokenizer, &device, args.batch_size)?;
    
    // Setup optimizer
    let mut params = Vec::new();
    // This would need to be adapted to get actual parameters from the model
    // In a real implementation, you would collect all trainable parameters
    let adam_params = ParamsAdamW {
        lr: args.learning_rate,
        ..Default::default()
    };
    // For simplicity, this is a placeholder
    // You would need to implement getting parameters from your BertClassifier
    let mut optimizer = AdamW::new(params, adam_params)?;
    
    // Train the model
    println!("Starting training...");
    println!("Training for {} epochs with batch size {}", args.epochs, args.batch_size);
    train_model(
        &mut classifier,
        &training_data,
        &mut optimizer,
        args.epochs,
        args.batch_size,
        &device,
    )?;
    
    // Save the trained model if path is provided
    if let Some(path) = args.save_checkpoint_path {
        classifier.save(Path::new(&path))?;
        println!("Model saved to {}", path);
    }
    
    // Optional: Run inference with the trained model
    println!("Running inference with trained model...");
    let test_sentences = [
        "The movie was terrible",
        "I really enjoyed this book",
    ];
    
    // Tokenize test sentences
    let tokens = tokenizer
        .encode_batch(test_sentences.to_vec(), true)
        .map_err(Error::msg)?;
    
    let token_ids = tokens
        .iter()
        .map(|tokens| {
            let tokens = tokens.get_ids().to_vec();
            Ok(Tensor::new(tokens.as_slice(), &device)?)
        })
        .collect::<Result<Vec<_>>>()?;
    
    let attention_mask = tokens
        .iter()
        .map(|tokens| {
            let tokens = tokens.get_attention_mask().to_vec();
            Ok(Tensor::new(tokens.as_slice(), &device)?)
        })
        .collect::<Result<Vec<_>>>()?;

    let token_ids = Tensor::stack(&token_ids, 0)?;
    let attention_mask = Tensor::stack(&attention_mask, 0)?;
    let token_type_ids = token_ids.zeros_like()?;
    
    // Run inference
    let logits = classifier.forward(&token_ids, &token_type_ids, Some(&attention_mask))?;
    
    // Get predictions
    let predictions = logits.argmax(1)?;
    let pred_values = predictions.to_vec1::<u8>()?;
    
    // Print results
    for (i, &pred) in pred_values.iter().enumerate() {
        println!("Sentence: '{}', Prediction: {}", test_sentences[i], pred);
    }
    
    Ok(())
}

pub fn normalize_l2(v: &Tensor) -> Result<Tensor> {
    Ok(v.broadcast_div(&v.sqr()?.sum_keepdim(1)?.sqrt()?)?)
}
