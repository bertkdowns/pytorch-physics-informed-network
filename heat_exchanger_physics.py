import torch

def heat_exchanger_residual(x, y):
    """
    Physics residual for heat exchanger energy balance.
    This function enforces the energy balance in a heat exchanger
    by calculating the difference between the energy lost by the hot fluid
    and the energy gained by the cold fluid.
    
    Args:
        x: Input tensor [batch_size, 4] with:
           x[:, 0] = m_dot_hot (mass flow rate of hot fluid)
           x[:, 1] = m_dot_cold (mass flow rate of cold fluid)
           x[:, 2] = T_hot_in (inlet temperature of hot fluid)
           x[:, 3] = T_cold_in (inlet temperature of cold fluid)
        
        y: Model prediction [batch_size, 2] with:
           y[:, 0] = T_hot_out (outlet temperature of hot fluid)
           y[:, 1] = T_cold_out (outlet temperature of cold fluid)
    """
    # Unpack input variables
    m_dot_hot = x[:, 0:1]    # Mass flow rate of hot fluid
    m_dot_cold = x[:, 1:2]   # Mass flow rate of cold fluid
    T_hot_in = x[:, 2:3]     # Inlet temperature of hot fluid
    T_cold_in = x[:, 3:4]    # Inlet temperature of cold fluid
    
    # Unpack predictions - outlet temperatures
    if y.shape[1] > 1:  # If model predicts both outlet temperatures
        T_hot_out = y[:, 0:1]
        T_cold_out = y[:, 1:2]
    else:  # If model only predicts one temperature
        T_hot_out = y  # Assuming it predicts hot outlet temperature
        # You'd need another model or relation for T_cold_out in this case
        raise ValueError("Model must predict both outlet temperatures")
    
    # Physical constants
    Cp_hot = 4.184   # Specific heat capacity of water [kJ/(kg·K)]
    Cp_cold = 4.184  # Specific heat capacity of water [kJ/(kg·K)]
    
    # Energy balance calculation
    Q_hot = m_dot_hot * Cp_hot * (T_hot_in - T_hot_out)     # Energy lost by hot fluid
    Q_cold = m_dot_cold * Cp_cold * (T_cold_out - T_cold_in)  # Energy gained by cold fluid
    
    # Residual: difference between energy lost and gained (should be 0)
    residual = Q_hot - Q_cold
    
    return residual