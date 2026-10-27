"""
Simplified SolarFlareNet Implementation
Based on: https://www.nature.com/articles/s41598-023-40884-1

This is a working implementation that integrates with your existing notebook.
Architecture: Conv1D + LSTM + Transformer Encoder Blocks
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np

class TransformerEncoderBlock(layers.Layer):
    """
    Transformer Encoder Block (TEB)
    Components: Dropout -> Multi-Head Attention -> BatchNorm -> Conv1D -> LSTM
    """
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1, **kwargs):
        super(TransformerEncoderBlock, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        
        # Multi-head attention
        self.att = layers.MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=embed_dim // num_heads,
            dropout=dropout_rate
        )
        
        # Batch normalization
        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()
        
        # Conv1D layer
        self.conv1d = layers.Conv1D(
            filters=ff_dim,
            kernel_size=1,
            activation='relu'
        )
        
        # LSTM layer
        self.lstm = layers.LSTM(
            ff_dim,
            return_sequences=True,
            dropout=dropout_rate
        )
        
        # Dropout layers
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
        
        # Projection layer to match dimensions
        self.projection = layers.Dense(embed_dim)
        
    def call(self, inputs, training=None):
        # Multi-head attention with residual connection
        attn_output = self.att(inputs, inputs, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.bn1(inputs + attn_output, training=training)
        
        # Conv1D
        conv_output = self.conv1d(out1)
        
        # LSTM
        lstm_output = self.lstm(conv_output, training=training)
        lstm_output = self.dropout2(lstm_output, training=training)
        
        # Project back to embed_dim if needed
        if lstm_output.shape[-1] != self.embed_dim:
            lstm_output = self.projection(lstm_output)
        
        # Residual connection and batch norm
        out2 = self.bn2(out1 + lstm_output, training=training)
        
        return out2
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout_rate": self.dropout_rate,
        })
        return config


def create_solarflarenet(
    input_shape,
    num_transformer_blocks=4,
    embed_dim=64,
    num_heads=4,
    ff_dim=64,
    dropout_rate=0.3,
    use_simple_version=False
):
    """
    Create SolarFlareNet model
    
    Parameters:
    -----------
    input_shape : tuple
        Shape of input data (timesteps, features) or (features,) for non-sequential
    num_transformer_blocks : int
        Number of transformer encoder blocks (default: 4, as in paper)
    embed_dim : int
        Embedding dimension
    num_heads : int
        Number of attention heads
    ff_dim : int
        Feed-forward dimension
    dropout_rate : float
        Dropout rate
    use_simple_version : bool
        If True, use a simpler model for faster training/testing
    
    Returns:
    --------
    keras.Model
    """
    
    # Handle both sequential and non-sequential data
    if len(input_shape) == 1:
        # Non-sequential data - reshape to (1, features)
        inputs = layers.Input(shape=input_shape)
        x = layers.Reshape((1, input_shape[0]))(inputs)
    else:
        # Sequential data
        inputs = layers.Input(shape=input_shape)
        x = inputs
    
    # Initial batch normalization
    x = layers.BatchNormalization()(x)
    
    if use_simple_version:
        # Simplified version for faster training
        x = layers.Conv1D(filters=32, kernel_size=1, activation='relu')(x)
        x = layers.LSTM(64, return_sequences=True, dropout=dropout_rate)(x)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(dropout_rate)(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
    else:
        # Full SolarFlareNet architecture
        # Initial Conv1D layer
        x = layers.Conv1D(
            filters=embed_dim,
            kernel_size=1,
            activation='relu',
            padding='same'
        )(x)
        x = layers.Dropout(dropout_rate)(x)
        
        # Stack of Transformer Encoder Blocks
        for i in range(num_transformer_blocks):
            x = TransformerEncoderBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout_rate=dropout_rate,
                name=f'transformer_block_{i}'
            )(x)
        
        # Global average pooling
        x = layers.GlobalAveragePooling1D()(x)
        
        # Classification head
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(dropout_rate)(x)
        
        # Output layer
        outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='SolarFlareNet')
    
    return model


def create_solarflarenet_for_tabular(
    num_features,
    dropout_rate=0.3,
    hidden_units=[128, 64, 32]
):
    """
    Simplified version specifically for tabular data (no time series)
    This is more appropriate when you don't have temporal sequences
    
    Parameters:
    -----------
    num_features : int
        Number of input features
    dropout_rate : float
        Dropout rate
    hidden_units : list
        List of hidden layer sizes
    
    Returns:
    --------
    keras.Model
    """
    inputs = layers.Input(shape=(num_features,))
    
    x = layers.BatchNormalization()(inputs)
    
    for units in hidden_units:
        x = layers.Dense(units, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
    
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='SolarFlareNet_Tabular')
    
    return model


if __name__ == "__main__":
    # Test the model creation
    print("Testing SolarFlareNet model creation...")
    
    # Test with sequential data
    print("\n1. Sequential data model:")
    model_seq = create_solarflarenet(
        input_shape=(10, 25),  # 10 timesteps, 25 features
        use_simple_version=False
    )
    model_seq.summary()
    
    # Test with tabular data
    print("\n2. Tabular data model:")
    model_tab = create_solarflarenet_for_tabular(num_features=25)
    model_tab.summary()
    
    print("\n✓ Model creation tests passed!")
