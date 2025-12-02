from pyexpat import model
import requests
import json
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime


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

    elif 'transect' in path_parts:
        # Layer query: /simulations/transect/model/lake/start/end/lat/lon
        lake = path_parts[3] if len(path_parts) > 3 else 'unknown'
        start_time = path_parts[4] if len(path_parts) > 4 else 'unknown'
        end_time = path_parts[5] if len(path_parts) > 5 else 'unknown'
        lat = path_parts[6] if len(path_parts) > 6 else 'unknown'
        lon = path_parts[7] if len(path_parts) > 7 else 'unknown'
        model = path_parts[2] if len(path_parts) > 2 else 'D3D'
        
        # Create filename
        filename = f"{lake}_{start_time}_{end_time}_{lat}_{lon}_{model}.json"
    
    #else:
    #    # Generic fallback
    #    filename = "alplakes_data.json"
    
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



def fetch_and_save_alplakes_data_transect(lake, start_time, end_time, lat, lon, variables=['temperature', 'velocity'], model='delft3d-flow'):

    lat_str = '%2C'.join([f'{var}' for var in lat])
    lon_str = '%2C'.join([f'{var}' for var in lon])

    variables_str = '&'.join([f'variables={var}' for var in variables])

    api_url = (f"https://alplakes-api.eawag.ch/simulations/transect/{model}/{lake}/"f"{start_time}/{end_time}/{lat_str}/{lon_str}?{variables_str}")
    output_file = generate_output_path(api_url)
    response = requests.get(api_url, timeout=30)
    data = json.loads(response.text)

    return data





def _find_time_index(time_list, time_input):
    """
    Find the index of a time in the time list.
    
    Parameters:
    -----------
    time_list : list
        List of time strings from the data
    time_input : str, datetime, int, or None
        The time to find
    
    Returns:
    --------
    int : The index of the matching time
    """
    # If None, return first index
    if time_input is None:
        return 0
    
    # If integer, use directly as index
    if isinstance(time_input, int):
        if time_input < 0:
            return len(time_list) + time_input  # Handle negative indexing
        return time_input
    
    # Convert input to datetime for comparison
    if isinstance(time_input, str):
        # Handle various string formats
        time_input = time_input.replace('Z', '+00:00')  # Handle Z timezone
        # Try parsing with different formats
        try:
            target_dt = datetime.fromisoformat(time_input)
        except:
            # Try without timezone
            try:
                target_dt = datetime.strptime(time_input, '%Y-%m-%d %H:%M:%S')
            except:
                try:
                    target_dt = datetime.strptime(time_input, '%Y-%m-%d %H:%M')
                except:
                    try:
                        target_dt = datetime.strptime(time_input, '%Y-%m-%dT%H:%M:%S')
                    except:
                        raise ValueError(f"Could not parse time string: {time_input}")
    elif isinstance(time_input, datetime):
        target_dt = time_input
    else:
        raise ValueError(f"time must be str, datetime, int, or None, got {type(time_input)}")
    
    # Convert all times in list to datetime and find closest match
    time_dts = []
    for t_str in time_list:
        t_str = t_str.replace('Z', '+00:00')
        try:
            time_dts.append(datetime.fromisoformat(t_str))
        except:
            time_dts.append(datetime.strptime(t_str, '%Y-%m-%dT%H:%M:%S'))
    
    # Make target_dt timezone-aware if comparing with timezone-aware times
    if time_dts and time_dts[0].tzinfo is not None and target_dt.tzinfo is None:
        # Make target timezone aware (assume UTC)
        from datetime import timezone
        target_dt = target_dt.replace(tzinfo=timezone.utc)
    elif time_dts and time_dts[0].tzinfo is None and target_dt.tzinfo is not None:
        # Make time_dts timezone aware
        from datetime import timezone
        time_dts = [t.replace(tzinfo=timezone.utc) if t.tzinfo is None else t for t in time_dts]
    
    # Find exact match or closest time
    if target_dt in time_dts:
        return time_dts.index(target_dt)
    
    # Find closest time
    time_diffs = [abs((t - target_dt).total_seconds()) for t in time_dts]
    closest_idx = time_diffs.index(min(time_diffs))
    
    # Warn if not exact match
    print(f"Note: Exact time not found. Using closest time: {time_list[closest_idx]}")
    
    return closest_idx

def plot_alplakes_transect(data, time=None, figsize=(16, 8), 
                          plot_temp=True, plot_velocity=True, cmap='jet',
                          quiver_scale=50, quiver_width=0.002, alpha_quiver=0.6):
    """
    Plot lake temperature and/or velocity for a transect cross-section
    
    Parameters:
    -----------
    data : dict
        The JSON data from Alplakes transect API
    time : str, datetime, or int, optional
        Which time to plot. Can be:
        - String in ISO format: '2025-10-23T15:00:00' or '2025-10-23 15:00'
        - datetime object
        - Integer index (0 for first, -1 for last)
        - None (default): uses first time step
    figsize : tuple, optional
        Figure size in inches (default: (16, 8))
    plot_temp : bool, optional
        Whether to plot temperature (default: True)
    plot_velocity : bool, optional
        Whether to plot velocity arrows (default: True)
    cmap : str, optional
        Colormap for temperature (default: 'jet')
        Other options: 'viridis', 'plasma', 'turbo', 'coolwarm', 'RdYlBu_r'
    quiver_scale : float, optional
        Scale for velocity arrows - larger values = smaller arrows (default: 50)
        Try values between 20-100 depending on velocity magnitude
    quiver_width : float, optional
        Width of velocity arrows (default: 0.002)
    alpha_quiver : float, optional
        Transparency of velocity arrows (default: 0.6)
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    
    Examples:
    ---------
    >>> import json
    >>> with open('transect_data.json', 'r') as f:
    ...     data = json.load(f)
    
    # Plot first time step (default)
    >>> fig, ax = plot_alplakes_transect(data)
    
    # Plot specific date/time
    >>> fig, ax = plot_alplakes_transect(data, time='2025-10-23T15:00:00')
    >>> fig, ax = plot_alplakes_transect(data, time='2025-10-23 15:00')
    
    # Plot using index
    >>> fig, ax = plot_alplakes_transect(data, time=0)  # First
    >>> fig, ax = plot_alplakes_transect(data, time=-1)  # Last
    
    >>> plt.show()
    """
    
    # Find the time index
    time_index = _find_time_index(data['time'], time)
    
    # Extract data
    distance = np.array(data['distance']['data'], dtype=float)  # Distance along transect
    depth = np.array(data['depth']['data'], dtype=float)  # Depth levels
    time_str = data['time'][time_index]  # Time string for this snapshot
    lat = np.array(data['lat'], dtype=float)
    lng = np.array(data['lng'], dtype=float)
    
    # Get the 3D data arrays [time, depth, distance]
    temp_3d = np.array(data['variables']['temperature']['data'], dtype=float)
    u_3d = np.array(data['variables']['u']['data'], dtype=float)  # Eastward velocity
    v_3d = np.array(data['variables']['v']['data'], dtype=float)  # Northward velocity
    
    # Extract the 2D slice for this time step [depth, distance]
    temp_2d = temp_3d[time_index, :, :]
    u_2d = u_3d[time_index, :, :]
    v_2d = v_3d[time_index, :, :]
    
    # Convert depth to negative values (for plotting with depth on y-axis)
    depth_negative = -np.abs(depth)
    
    # Create meshgrid for plotting
    Distance, Depth = np.meshgrid(distance, depth_negative)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot temperature as pcolormesh if requested
    if plot_temp:
        # Mask out NaN values for better visualization
        temp_masked = np.ma.masked_invalid(temp_2d)
        
        mesh = ax.pcolormesh(Distance/1000, Depth, temp_masked, 
                            cmap=cmap, shading='auto')
        
        # Add colorbar
        cbar = plt.colorbar(mesh, ax=ax, orientation='vertical', 
                           pad=0.02, aspect=30)
        cbar.set_label('Temperature (°C)', fontsize=12)
    
    # Plot velocity vectors if requested
    if plot_velocity:
        # Calculate velocity magnitude for the transect
        # For a vertical transect, we need the component perpendicular to the transect
        # and the vertical component (if available)
        
        # Calculate horizontal velocity magnitude
        vel_horizontal = np.sqrt(u_2d**2 + v_2d**2)
        
        # For quiver plot, subsample to avoid overcrowding
        skip_dist = max(1, len(distance) // 30)  # Show ~30 arrows horizontally
        skip_depth = max(1, len(depth) // 20)     # Show ~20 arrows vertically
        
        # Create subsampled arrays
        Distance_sub = Distance[::skip_depth, ::skip_dist]
        Depth_sub = Depth[::skip_depth, ::skip_dist]
        u_sub = u_2d[::skip_depth, ::skip_dist]
        v_sub = v_2d[::skip_depth, ::skip_dist]
        
        # Calculate the angle of the transect to get proper velocity components
        # Transect direction (approximate as linear for velocity projection)
        delta_lat = lat[-1] - lat[0]
        delta_lng = lng[-1] - lng[0]
        transect_angle = np.arctan2(delta_lat, delta_lng)
        
        # Project velocity onto transect direction (along-transect component)
        # Positive = flow in direction of transect
        vel_along = u_sub * np.cos(transect_angle) + v_sub * np.sin(transect_angle)
        
        # For now, we don't have vertical velocity, so use zeros
        # If vertical velocity data exists in future, it can be added here
        vel_vertical = np.zeros_like(vel_along)
        
        # Filter out NaN values
        valid = ~(np.isnan(Distance_sub) | np.isnan(Depth_sub) | 
                 np.isnan(vel_along) | np.isnan(vel_vertical))
        
        # Plot velocity arrows
        quiver = ax.quiver(Distance_sub[valid]/1000, Depth_sub[valid], 
                          vel_along[valid], vel_vertical[valid],
                          scale=quiver_scale, width=quiver_width,
                          color='black', alpha=alpha_quiver,
                          headwidth=3, headlength=4)
        
        # Calculate reference velocity for quiver key
        vel_mag = np.sqrt(vel_along[valid]**2 + vel_vertical[valid]**2)
        ref_velocity = np.round(np.nanmax(vel_mag) * 0.15, 2)
        if ref_velocity == 0 or np.isnan(ref_velocity):
            ref_velocity = 0.1
        
        # Add quiver key
        ax.quiverkey(quiver, 0.85, 0.95, ref_velocity, 
                    f'{ref_velocity} m/s', 
                    labelpos='E', coordinates='figure',
                    color='black', labelcolor='black')
    
    # Add reference line at surface (depth = 0)
    ax.axhline(y=0, color='darkblue', linestyle='--', linewidth=1.5, 
               alpha=0.7, label='Surface')
    
    # Labels and formatting
    ax.set_xlabel('Distance along transect (km)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Depth (m)', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, linestyle=':', linewidth=0.5)
    
    # Format time string for title
    try:
        dt = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
        time_formatted = dt.strftime('%Y-%m-%d %H:%M UTC')
    except:
        time_formatted = time_str
    
    # Dynamic title based on what's plotted
    title_parts = []
    if plot_temp:
        title_parts.append('Temperature')
    if plot_velocity:
        title_parts.append('Currents')
    
    title = f"Lake Transect - {' & '.join(title_parts)}\n{time_formatted}"
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    
    # Add transect endpoints info as text
    coord_text = f"From: ({lat[0]:.4f}°, {lng[0]:.4f}°) → To: ({lat[-1]:.4f}°, {lng[-1]:.4f}°)"
    ax.text(0.5, -0.12, coord_text, transform=ax.transAxes,
           ha='center', fontsize=9, style='italic', color='gray')
    
    # Set y-axis limits to show only where we have data
    max_depth = np.nanmin(depth_negative)  # Most negative = deepest
    ax.set_ylim(max_depth * 1.05, 5)  # Add 5% margin and go slightly above surface
    
    plt.tight_layout()
    return fig, ax


def OLDplot_alplakes_transect_timeseries(data, figsize=(18, 10), 
                                     plot_temp=True, cmap='jet',
                                     levels=20, time_range=None):
    """
    Plot temperature evolution over time as a time-depth contour plot
    
    Parameters:
    -----------
    data : dict
        The JSON data from Alplakes transect API
    figsize : tuple, optional
        Figure size in inches (default: (18, 10))
    plot_temp : bool, optional
        Whether to plot temperature (default: True)
    cmap : str, optional
        Colormap for temperature (default: 'jet')
    levels : int, optional
        Number of contour levels (default: 20)
    time_range : tuple of (start, end), optional
        Time range to plot. Each can be:
        - String in ISO format: '2025-10-23T15:00:00'
        - datetime object
        - Integer index
        - None (uses all times)
        Example: ('2025-10-23T15:00', '2025-10-24T06:00')
    
    Returns:
    --------
    fig, axes : matplotlib figure and axes objects
    
    Examples:
    ---------
    >>> # Plot all time steps
    >>> fig, axes = plot_alplakes_transect_timeseries(data)
    
    >>> # Plot specific time range
    >>> fig, axes = plot_alplakes_transect_timeseries(
    ...     data, 
    ...     time_range=('2025-10-23T15:00', '2025-10-24T06:00')
    ... )
    
    >>> plt.show()
    """
    
    # Extract data
    times = data['time']
    distance = np.array(data['distance']['data'], dtype=float)
    depth = np.array(data['depth']['data'], dtype=float)
    
    # Filter time range if specified
    if time_range is not None:
        start_time, end_time = time_range
        start_idx = _find_time_index(times, start_time) if start_time is not None else 0
        end_idx = _find_time_index(times, end_time) if end_time is not None else len(times) - 1
        
        # Create filtered lists
        time_indices = range(start_idx, end_idx + 1)
        times = [times[i] for i in time_indices]
    else:
        time_indices = range(len(times))
    
    # Get the 3D temperature data [time, depth, distance]
    temp_3d = np.array(data['variables']['temperature']['data'], dtype=float)
    
    # Convert depth to negative values
    depth_negative = -np.abs(depth)
    
    # Create figure with subplots for each time step
    n_times = len(times)
    n_cols = 3
    n_rows = int(np.ceil(n_times / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, 
                            sharex=True, sharey=True)
    axes = axes.flatten() if n_times > 1 else [axes]
    
    # Get global min/max for consistent color scale
    temp_min = np.nanmin(temp_3d)
    temp_max = np.nanmax(temp_3d)
    
    # Create meshgrid
    Distance, Depth = np.meshgrid(distance, depth_negative)
    
    for plot_idx, time_idx in enumerate(time_indices):
        if plot_idx >= n_times:
            axes[plot_idx].set_visible(False)
            continue
            
        ax = axes[plot_idx]
        time_str = times[plot_idx]
        
        # Get 2D slice for this time
        temp_2d = temp_3d[time_idx, :, :]
        temp_masked = np.ma.masked_invalid(temp_2d)
        
        # Plot
        mesh = ax.pcolormesh(Distance/1000, Depth, temp_masked, 
                           cmap=cmap, shading='auto',
                           vmin=temp_min, vmax=temp_max)
        
        # Format time
        try:
            dt = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
            time_formatted = dt.strftime('%m-%d %H:%M')
        except:
            time_formatted = time_str[:16]
        
        ax.set_title(time_formatted, fontsize=10, fontweight='bold')
        ax.grid(alpha=0.3, linestyle=':', linewidth=0.5)
        ax.axhline(y=0, color='darkblue', linestyle='--', 
                  linewidth=1, alpha=0.5)
        
        # Labels only on left and bottom
        if plot_idx % n_cols == 0:
            ax.set_ylabel('Depth (m)', fontsize=10)
        if plot_idx >= n_times - n_cols:
            ax.set_xlabel('Distance (km)', fontsize=10)
    
    # Add single colorbar for all subplots
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(mesh, cax=cbar_ax)
    cbar.set_label('Temperature (°C)', fontsize=12, fontweight='bold')
    
    # Overall title
    lat = np.array(data['lat'], dtype=float)
    lng = np.array(data['lng'], dtype=float)
    fig.suptitle(f'Lake Transect Temperature Evolution\n' + 
                f'From ({lat[0]:.4f}°, {lng[0]:.4f}°) → ' +
                f'To ({lat[-1]:.4f}°, {lng[-1]:.4f}°)',
                fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 0.92, 0.96])
    return fig, axes










def plot_alplakes_transect_timeseries(data, figsize=(18, 10), 
                                     plot_temp=True, cmap='jet',
                                     levels=20, time_range=None, time_step=None):
    """
    Plot temperature evolution over time as a time-depth contour plot
    
    Parameters:
    -----------
    data : dict
        The JSON data from Alplakes transect API
    figsize : tuple, optional
        Figure size in inches (default: (18, 10))
    plot_temp : bool, optional
        Whether to plot temperature (default: True)
    cmap : str, optional
        Colormap for temperature (default: 'jet')
    levels : int, optional
        Number of contour levels (default: 20)
    time_range : tuple of (start, end), optional
        Time range to plot. Each can be:
        - String in ISO format: '2025-10-23T15:00:00'
        - datetime object
        - Integer index
        - None (uses all times)
        Example: ('2025-10-23T15:00', '2025-10-24T06:00')
    time_step : int, float, or str, optional
        Time interval between plots. Can be:
        - Integer: number of time steps to skip (e.g., 2 = every other time)
        - Float: hours between plots (e.g., 24.0 = every 24 hours)
        - String: '3h', '6h', '12h', '24h' (hours between plots)
        - None: use all available times (default)
        Example: time_step=24.0 for daily intervals
    
    Returns:
    --------
    fig, axes : matplotlib figure and axes objects
    
    Examples:
    ---------
    >>> # Plot all time steps
    >>> fig, axes = plot_alplakes_transect_timeseries(data)
    
    >>> # Plot specific time range
    >>> fig, axes = plot_alplakes_transect_timeseries(
    ...     data, 
    ...     time_range=('2025-10-23T15:00', '2025-10-24T06:00')
    ... )
    
    >>> # Plot every 24 hours
    >>> fig, axes = plot_alplakes_transect_timeseries(data, time_step=24.0)
    >>> fig, axes = plot_alplakes_transect_timeseries(data, time_step='24h')
    
    >>> # Plot every 6 hours
    >>> fig, axes = plot_alplakes_transect_timeseries(data, time_step='6h')
    
    >>> plt.show()
    """
    
    # Extract data
    times = data['time']
    distance = np.array(data['distance']['data'], dtype=float)
    depth = np.array(data['depth']['data'], dtype=float)
    
    # Filter time range if specified
    if time_range is not None:
        start_time, end_time = time_range
        start_idx = _find_time_index(times, start_time) if start_time is not None else 0
        end_idx = _find_time_index(times, end_time) if end_time is not None else len(times) - 1
        
        # Create filtered lists
        time_indices = list(range(start_idx, end_idx + 1))
    else:
        time_indices = list(range(len(times)))
    
    # Parse time_step parameter
    if time_step is not None:
        if isinstance(time_step, str):
            # Parse string like '3h', '6h', '24h'
            if time_step.endswith('h'):
                target_hours = float(time_step[:-1])
            else:
                raise ValueError(f"time_step string must end with 'h', got: {time_step}")
        elif isinstance(time_step, (int, float)):
            if time_step >= 1 and time_step == int(time_step) and time_step < 10:
                # Small integer - treat as index step
                time_indices = time_indices[::int(time_step)]
                target_hours = None
            else:
                # Float or large number - treat as hours
                target_hours = float(time_step)
        else:
            raise ValueError(f"time_step must be int, float, or str, got {type(time_step)}")
        
        # If we have target hours, filter by time intervals
        if time_step is not None and isinstance(time_step, (float, str)):
            # Convert times to datetime objects
            time_dts = []
            for idx in time_indices:
                t_str = times[idx].replace('Z', '+00:00')
                try:
                    time_dts.append(datetime.fromisoformat(t_str))
                except:
                    time_dts.append(datetime.strptime(t_str, '%Y-%m-%dT%H:%M:%S'))
            
            # Filter to get roughly the desired interval
            if len(time_dts) > 0:
                filtered_indices = [time_indices[0]]  # Always include first
                last_time = time_dts[0]
                
                for i, (idx, dt) in enumerate(zip(time_indices[1:], time_dts[1:]), 1):
                    hours_diff = abs((dt - last_time).total_seconds() / 3600)
                    if hours_diff >= target_hours * 0.9:  # 90% tolerance
                        filtered_indices.append(idx)
                        last_time = dt
                
                time_indices = filtered_indices
    
    # Get filtered time strings
    times_filtered = [times[i] for i in time_indices]
    
    # Get the 3D temperature data [time, depth, distance]
    temp_3d = np.array(data['variables']['temperature']['data'], dtype=float)
    
    # Convert depth to negative values
    depth_negative = -np.abs(depth)
    
    # Create figure with subplots for each time step
    n_times = len(times_filtered)
    n_cols = 3
    n_rows = int(np.ceil(n_times / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, 
                            sharex=True, sharey=True)
    axes = axes.flatten() if n_times > 1 else [axes]
    
    # Get global min/max for consistent color scale
    temp_min = np.nanmin(temp_3d)
    temp_max = np.nanmax(temp_3d)
    
    # Create meshgrid
    Distance, Depth = np.meshgrid(distance, depth_negative)
    
    # Calculate time deltas for display
    time_deltas = []
    if n_times > 1:
        for i in range(len(time_indices)):
            if i == 0:
                time_deltas.append(None)  # First plot has no delta
            else:
                # Calculate time difference from previous plot
                t1_str = times[time_indices[i-1]].replace('Z', '+00:00')
                t2_str = times[time_indices[i]].replace('Z', '+00:00')
                try:
                    t1 = datetime.fromisoformat(t1_str)
                    t2 = datetime.fromisoformat(t2_str)
                except:
                    t1 = datetime.strptime(t1_str, '%Y-%m-%dT%H:%M:%S')
                    t2 = datetime.strptime(t2_str, '%Y-%m-%dT%H:%M:%S')
                
                delta_seconds = (t2 - t1).total_seconds()
                delta_hours = delta_seconds / 3600
                
                # Format delta nicely
                if delta_hours >= 24:
                    delta_str = f"+{delta_hours/24:.1f}d"
                elif delta_hours >= 1:
                    delta_str = f"+{delta_hours:.1f}h"
                else:
                    delta_str = f"+{delta_hours*60:.0f}min"
                
                time_deltas.append(delta_str)
    
    for plot_idx, time_idx in enumerate(time_indices):
        if plot_idx >= n_times:
            axes[plot_idx].set_visible(False)
            continue
            
        ax = axes[plot_idx]
        time_str = times_filtered[plot_idx]
        
        # Get 2D slice for this time
        temp_2d = temp_3d[time_idx, :, :]
        temp_masked = np.ma.masked_invalid(temp_2d)
        
        # Plot
        mesh = ax.pcolormesh(Distance/1000, Depth, temp_masked, 
                           cmap=cmap, shading='auto',
                           vmin=temp_min, vmax=temp_max)
        
        # Format time
        try:
            dt = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
            time_formatted = dt.strftime('%m-%d %H:%M')
        except:
            time_formatted = time_str[:16]
        
        # Add time delta to title if not first plot
        if time_deltas and plot_idx < len(time_deltas) and time_deltas[plot_idx]:
            title_text = f'{time_formatted}\n({time_deltas[plot_idx]})'
        else:
            title_text = time_formatted
        
        ax.set_title(title_text, fontsize=10, fontweight='bold')
        ax.grid(alpha=0.3, linestyle=':', linewidth=0.5)
        ax.axhline(y=0, color='darkblue', linestyle='--', 
                  linewidth=1, alpha=0.5)
        
        # Labels only on left and bottom
        if plot_idx % n_cols == 0:
            ax.set_ylabel('Depth (m)', fontsize=10)
        if plot_idx >= n_times - n_cols:
            ax.set_xlabel('Distance (km)', fontsize=10)
    
    # Add single colorbar for all subplots
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(mesh, cax=cbar_ax)
    cbar.set_label('Temperature (°C)', fontsize=12, fontweight='bold')
    
    # Overall title
    lat = np.array(data['lat'], dtype=float)
    lng = np.array(data['lng'], dtype=float)
    fig.suptitle(f'Lake Transect Temperature Evolution\n' + 
                f'From ({lat[0]:.4f}°, {lng[0]:.4f}°) → ' +
                f'To ({lat[-1]:.4f}°, {lng[-1]:.4f}°)',
                fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 0.92, 0.96])
    return fig, axes

