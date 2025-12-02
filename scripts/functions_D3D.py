from pyexpat import model
import requests
import json
import numpy as np
import matplotlib.pyplot as plt
import os


def generate_output_path(api_url, base_dir='../data'):
    """
    Generate output file path from API URL.
    
    Parameters:
    -----------
    api_url : str
        The API URL
    base_dir : str, optional
        Base directory for output files (default: '../data')
    
    Returns:
    --------
    str
        Output file path
    """
    import re
    from urllib.parse import urlparse, parse_qs
    
    # Parse the URL
    parsed = urlparse(api_url)
    path_parts = parsed.path.strip('/').split('/')
    
    # Extract relevant parts
    if 'point' in path_parts:
        # Point query: /simulations/point/model/lake/start/end/depth/lat/lon
        lake = path_parts[3] if len(path_parts) > 3 else 'unknown'
        start_time = path_parts[4] if len(path_parts) > 4 else 'unknown'
        end_time = path_parts[5] if len(path_parts) > 5 else 'unknown'
        depth = path_parts[6] if len(path_parts) > 6 else 'unknown'
        
        # Create filename
        filename = f"{lake}_point_{start_time}_{end_time}_{depth}m.json"
        
    elif 'layer' in path_parts:
        # Layer query: /simulations/layer/model/lake/time/depth
        lake = path_parts[3] if len(path_parts) > 3 else 'unknown'
        time = path_parts[4] if len(path_parts) > 4 else 'unknown'
        depth = path_parts[5] if len(path_parts) > 5 else 'unknown'
        model = path_parts[2] if len(path_parts) > 2 else 'D3D'
        
        # Create filename
        filename = f"{lake}_{time}__{depth}m_{model}.json"
    
    else:
        # Generic fallback
        filename = "alplakes_data.json"
    
    # Create full path
    lake_dir = lake.capitalize() if 'lake' in locals() else 'Unknown'
    output_path = os.path.join(base_dir, lake_dir, filename)
    
    return output_path

# Usage example
# data = fetch_and_save_alplakes_point_data(
#     lake='geneva',
#     start_time='202304050300',
#     end_time='202304112300',
#     depth=1,
#     lat=46.5,
#     lon=6.67,
#     variables=['temperature', 'velocity']
# )



    


def fetch_and_save_alplakes_data_map(lake, date, depth, variables, model='delft3d-flow', verbose=True):

    """
    Fetch data from Alplakes API and save to JSON file.
    
    Parameters:
    -----------
    api_url : str
        The API URL to fetch data from
    verbose : bool, optional
        Whether to print progress messages (default: True)
    
    Returns:
    --------
    dict or None
        The fetched data if successful, None otherwise
    """
    # Construct API URL
    variables_str = '&'.join([f'variables={var}' for var in variables])
    api_url = (f"https://alplakes-api.eawag.ch/simulations/layer/{model}/{lake}/"f"{date}/{depth}?{variables_str}")
    
    output_file = generate_output_path(api_url)
    
    if verbose:
        print(f"Output file: {output_file}")
        print("Fetching data from API...")
    
    try:
        response = requests.get(api_url, timeout=30)
        
        # Check if request was successful
        if response.status_code == 200:
            if verbose:
                print("✓ Data fetched successfully!")
                print(f"  Response length: {len(response.text)} characters")
            
            # The response is JSON formatted, parse it
            data = json.loads(response.text)
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # Save to JSON file
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            if verbose:
                print(f"✓ Data saved to: {output_file}")
                
                # Print some basic info about the data
                print(f"\nData summary:")
                print(f"  Time: {data.get('time', 'N/A')}")
                print(f"  Depth: {data.get('depth', {}).get('data', 'N/A')} m")
                print(f"  Variables: {list(data.get('variables', {}).keys())}")
            
            return data
            
        else:
            if verbose:
                print(f"✗ Error: {response.status_code}")
                print(response.text)
            return None
            
    except requests.exceptions.RequestException as e:
        if verbose:
            print(f"✗ Network error: {e}")
            print("\nThis script needs to be run on your local machine with internet access.")
        return None


# Usage
# api_url = "https://alplakes-api.eawag.ch/simulations/layer/delft3d-flow/joux/202304050300/1?variables=temperature&variables=velocity"
# data = fetch_and_save_alplakes_data(api_url)


def plot_alplakes_pcolormesh(data, skip=5, figsize=(14, 10), 
                            plot_temp=True, plot_velocity=True, cmap='jet',
                            quiver_scale=10, quiver_width=0.003):
    """
    Plot lake temperature and/or velocity currents
    
    Parameters:
    -----------
    data : dict
        The JSON data from Alplakes API
    skip : int, optional
        Subsample velocity arrows - plot every nth point (default: 5)
    figsize : tuple, optional
        Figure size in inches (default: (14, 10))
    plot_temp : bool, optional
        Whether to plot temperature (default: True)
    plot_velocity : bool, optional
        Whether to plot velocity arrows (default: True)
    cmap : str, optional
        Colormap for temperature (default: 'jet')
        Other options: 'viridis', 'plasma', 'turbo', 'coolwarm', 'RdYlBu_r'
    quiver_scale : float, optional
        Scale for velocity arrows - larger values = smaller arrows (default: 10)
        Try values between 5-50 depending on velocity magnitude
    quiver_width : float, optional
        Width of velocity arrows (default: 0.003)
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """
    lat = np.array(data['lat'], dtype=float)
    lng = np.array(data['lng'], dtype=float)
    u = np.array(data['variables']['u']['data'], dtype=float)
    v = np.array(data['variables']['v']['data'], dtype=float)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Flatten all arrays
    lat_flat = lat.flatten()
    lng_flat = lng.flatten()
    u_flat = u.flatten()
    v_flat = v.flatten()
    
    # Plot temperature if requested
    if plot_temp:
        temp = np.array(data['variables']['temperature']['data'], dtype=float)
        temp_flat = temp.flatten()
        
        # Valid points for temperature
        valid_temp = ~(np.isnan(lat_flat) | np.isnan(lng_flat) | np.isnan(temp_flat))
        
        # Plot temperature as scatter with actual coordinates
        scatter = ax.scatter(lng_flat[valid_temp], lat_flat[valid_temp], 
                            c=temp_flat[valid_temp], s=50, cmap=cmap,
                            edgecolors='none', alpha=0.9)
        
        # Horizontal colorbar at the bottom
        cbar = plt.colorbar(scatter, ax=ax, orientation='horizontal', 
                           pad=0.08, shrink=0.8, aspect=30)
        cbar.set_label('Temperature (°C)', fontsize=12)
    
    # Plot velocity if requested
    if plot_velocity:
        # Valid points for velocity (with skip)
        valid_vel = ~(np.isnan(lat_flat) | np.isnan(lng_flat) | np.isnan(u_flat) | np.isnan(v_flat))
        
        lat_vel = lat_flat[valid_vel][::skip]
        lng_vel = lng_flat[valid_vel][::skip]
        u_vel = u_flat[valid_vel][::skip]
        v_vel = v_flat[valid_vel][::skip]
        
        # Calculate reference velocity for quiver key (10% of max velocity magnitude)
        vel_magnitude = np.sqrt(u_vel**2 + v_vel**2)
        ref_velocity = np.round(np.nanmax(vel_magnitude) * 0.1, 2)
        if ref_velocity == 0:
            ref_velocity = 0.1
        
        # Plot velocity arrows with user-defined scale
        quiver = ax.quiver(lng_vel, lat_vel, u_vel, v_vel,
                          scale=quiver_scale, color='black', alpha=0.7, 
                          width=quiver_width)
        
        # Add quiver key (scale reference) with dynamic reference velocity
        ax.quiverkey(quiver, 0.9, 0.95, ref_velocity, f'{ref_velocity} m/s', 
                    labelpos='E', coordinates='figure')
    
    # Labels and title
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_aspect('equal')
    ax.grid(alpha=0.3)
    
    # Dynamic title based on what's plotted
    title_parts = []
    if plot_temp:
        title_parts.append('Temperature')
    if plot_velocity:
        title_parts.append('Currents')
    
    title = f"Lake {' & '.join(title_parts)}\nDepth: {data['depth']['data']:.1f}m | {data['time']}"
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig, ax



def fetch_and_save_alplakes_point_data(lake, start_time, end_time, depth, lat, lon, 
                                       variables=['temperature', 'velocity'], 
                                       model='delft3d-flow', verbose=True):
    """
    Fetch point time-series data from Alplakes API and save to JSON file.
    
    Parameters:
    -----------
    lake : str
        Lake name (e.g., 'geneva', 'joux', 'constance')
    start_time : str
        Start time in format 'YYYYMMDDHHMM' (e.g., '202304050300')
    end_time : str
        End time in format 'YYYYMMDDHHMM' (e.g., '202304112300')
    depth : float or int
        Depth in meters (e.g., 1)
    lat : float
        Latitude coordinate (e.g., 46.5)
    lon : float
        Longitude coordinate (e.g., 6.67)
    variables : list of str, optional
        Variables to fetch (default: ['temperature', 'velocity'])
    model : str, optional
        Model name (default: 'delft3d-flow')
    verbose : bool, optional
        Whether to print progress messages (default: True)
    
    Returns:
    --------
    dict or None
        The fetched data if successful, None otherwise
    
    Example:
    --------
    >>> data = fetch_and_save_alplakes_point_data(
    ...     lake='geneva',
    ...     start_time='202304050300',
    ...     end_time='202304112300',
    ...     depth=1,
    ...     lat=46.5,
    ...     lon=6.67
    ... )
    """
    # Construct API URL
    variables_str = '&'.join([f'variables={var}' for var in variables])
    api_url = (f"https://alplakes-api.eawag.ch/simulations/point/{model}/{lake}/"
               f"{start_time}/{end_time}/{depth}/{lat}/{lon}?{variables_str}")
    
    # Generate output file path
    output_file = generate_output_path(api_url)
    
    if verbose:
        print(f"API URL: {api_url}")
        print(f"Output file: {output_file}")
        print("Fetching data from API...")
    
    try:
        response = requests.get(api_url, timeout=30)
        
        # Check if request was successful
        if response.status_code == 200:
            if verbose:
                print("✓ Data fetched successfully!")
                print(f"  Response length: {len(response.text)} characters")
            
            # The response is JSON formatted, parse it
            data = json.loads(response.text)
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # Save to JSON file
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            if verbose:
                print(f"✓ Data saved to: {output_file}")
                
                # Print some basic info about the data
                print(f"\nData summary:")
                print(f"  Time steps: {len(data.get('time', []))}")
                print(f"  Location: ({data.get('lat', 'N/A')}, {data.get('lng', 'N/A')})")
                print(f"  Depth: {data.get('depth', {}).get('data', 'N/A')} m")
                print(f"  Distance from requested point: {data.get('distance', {}).get('data', 'N/A')} m")
                print(f"  Variables: {list(data.get('variables', {}).keys())}")
                
                # Print time range
                if 'time' in data and len(data['time']) > 0:
                    print(f"  Time range: {data['time'][0]} to {data['time'][-1]}")
            
            return data
            
        else:
            if verbose:
                print(f"✗ Error: {response.status_code}")
                print(response.text)
            return None
            
    except requests.exceptions.RequestException as e:
        if verbose:
            print(f"✗ Network error: {e}")
            print("\nThis script needs to be run on your local machine with internet access.")
        return None

def plot_temperature_timeseries(data, figsize=(14, 6)):
    """Simple temperature time series plot."""
    import pandas as pd
    
    times = pd.to_datetime(data['time'])
    temperatures = data['variables']['temperature']['data']
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(times, temperatures, linewidth=2)
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Temperature (°C)', fontsize=12)
    ax.set_title(f'Water Temperature Time Series (Depth: {data["depth"]["data"]:.2f} m)', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    return fig, ax