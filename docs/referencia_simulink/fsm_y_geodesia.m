function ECEF_DOT   = fcn_get_R_n_e(ECEF,eta_dot)


p_dot=[eta_dot(1);
       eta_dot(2);
       eta_dot(3)];
l_rad=ECEF(1);
mu_rad=ECEF(2);

c_l  = cos(l_rad);
s_l  = sin(l_rad);
c_mu = cos(mu_rad);
s_mu = sin(mu_rad);

R_n_e = [ -c_l*s_mu,  -s_l,  -c_l*c_mu;
          -s_l*s_mu,   c_l,  -s_l*c_mu;
           c_mu,       0,    -s_mu     ];

ECEF_DOT=R_n_e*p_dot;       
%REC_NED_TEST=R_n_e'*deltaECEF;
end


function [lla_rad_vec, N,i,lla_deg_lon_lat] = fcn_ecef2lla(ecef_pos)
% INPUTS:
%   ecef_pos: 3-element vector [x, y, z] in meters
% OUTPUTS:
%   lla_rad_vec:     3-element vector [lon_rad, lat_rad, h] (rad, rad, m)
%   lla_deg_lat_lon: 2-element vector [lat_deg, lon_deg] (degrees)
%   depth_m:         Scalar depth (meters, positive down, -h)

% --- WGS-84 Ellipsoid Parameters ---
r_e = 6378137.0;       
e_sq = 0.00669437999014; 
% --- Extract inputs ---
x = ecef_pos(1);
y = ecef_pos(2);
z = ecef_pos(3);

%  Longitude (l) ---
lon_rad = atan2(y, x);

% latitude (mu) and Height (h) ---
p = sqrt(x^2 + y^2);
tol = 1e-7;
max_iter = 10;
mu_rad = atan(z / (p * (1.0 - e_sq))); 
h = 0.0;
N = 0.0;

for i = 1:max_iter
    mu_prev = mu_rad;
    sin_mu = sin(mu_rad);
    
    %  N 
    N = r_e / sqrt(1.0 - e_sq * sin_mu^2);
    
    %  height
    h = (p / cos(mu_rad)) - N;
    
    %  latitude
    mu_rad = atan(z / (p * (1.0 - e_sq * (N / (N + h)))));
    
    if abs(mu_rad - mu_prev) < tol
        break;
    end
end

h = (p / cos(mu_rad)) - N;

%  Outputs 
lla_rad_vec = [lon_rad; mu_rad; h];
lla_deg_lon_lat = [ lon_rad * (180.0/pi); mu_rad * (180.0/pi)];
%depth_m = -h; % Convert height (positive-up) to depth (positive-down)

end


function STOP  = fcn_Glider_State_Machine(mission_complete, t)
    persistent flag timesignal
    
    if isempty(flag)
        flag = 0;
        timesignal = 0;
    end

    STOP = 0;

    if mission_complete == 1 && flag == 0
        timesignal = t;
        flag = 1;
    end 
    
    if flag == 1 && (t - timesignal) > 20
        STOP = 1;
    end
end

function [pitch_ref,yaw_ref,Buoy_ref,mission_complete,target_xy_ned,debug_outputs,yaw_roll_coupling_gain]=...
    fcn_Glider_State_Machine(current_lla_fix,depth_sensor_m,current_yaw_rad,t)

acceptance_radius_m=90.0;
surface_wait_time_s=60.0;
bottom_wait_time_s=10.0;
pitch_antenna_rad=-1.1;
pitch_down_rad=-0.6109;
pitch_up_rad=0.6109;
buoy_down_ref=3;
buoy_up_ref=-3;
max_dive_depth_m=200.0;
depth_epsilon_m=2.0;
surface_epsilon_m=1;
initial_turn_depth_m=2.0;
flare_depth_m=3.5;
surface_stabilize_time_s=10.0;
min_estimation_depth_m=15.0;
 
pitch_flare_target=-1.1;       
flare_duration_s=10.0;          

STATE_FIX_PLAN=1;
STATE_SUBMERGE=2;
STATE_DIVE_TURN=3;
STATE_BOTTOM_WAIT=4;
STATE_CLIMB=5;
STATE_DONE=6;
STATE_SURFACE_WAIT=7;

persistent is_initialized
persistent ECEF_Origin
persistent R_e_n_Origin
persistent waypoints_ned
persistent num_waypoints
persistent current_dive_depth_m
persistent waypoints_lla_deg
persistent dive_start_ned
persistent max_depth_this_dive
persistent glide_ratio_k
persistent baseline_glide_ratio
persistent D_norm_filtered
persistent alpha_filter
persistent current_wp_index
persistent glider_state
persistent t_state_start
persistent current_yaw_ref
persistent yaw_ref_last_dive
persistent dist_to_wp_last_fix
persistent Dist_to_Target
persistent flare_active          % flag: flare has been triggered
persistent t_flare_start         % time when flare was triggered
persistent pitch_at_flare_entry  % pitch value when flare began

if isempty(is_initialized)
    is_initialized=1;
    current_wp_index=1;
    glider_state=STATE_FIX_PLAN;
    t_state_start=0.0;
    current_yaw_ref=0.0;
    current_dive_depth_m=max_dive_depth_m;
    dist_to_wp_last_fix=0.0;
    glide_ratio_k=1.4;
    Dist_to_Target=0;
    baseline_glide_ratio=1.4;
    D_norm_filtered=[0;0];
    alpha_filter=0.8;
    yaw_ref_last_dive=0.0;
    dive_start_ned=[0;0];
    max_depth_this_dive=0;
    flare_active=false;
    t_flare_start=0.0;
    pitch_at_flare_entry=pitch_up_rad;
    
        
     % Waypoints
    waypoints_lla_deg=[
    -95.72275,19.37214,0.0;
    -95.72011,19.37464,0.0;
    -95.72275,19.37714,0.0;
    ];





    num_waypoints=size(waypoints_lla_deg,1);

    LLA_Origin_deg=waypoints_lla_deg(1,:);
    LLA_Origin_rad=[LLA_Origin_deg(1)*(pi/180);
                    LLA_Origin_deg(2)*(pi/180);
                    LLA_Origin_deg(3)];

    ECEF_Origin=fcn_lla2ecef(LLA_Origin_rad);
    R_n_e_Origin=fcn_get_R_n_e_matrix(LLA_Origin_rad(1),LLA_Origin_rad(2));
    R_e_n_Origin=R_n_e_Origin';

    waypoints_ned=zeros(3,num_waypoints);
    for i=1:num_waypoints
        LLA_Target_deg=waypoints_lla_deg(i,:);
        LLA_Target_rad=[LLA_Target_deg(1)*(pi/180);
                        LLA_Target_deg(2)*(pi/180);
                        LLA_Target_deg(3)];
        ECEF_Target=fcn_lla2ecef(LLA_Target_rad);
        deltaECEF=ECEF_Target-ECEF_Origin;
        waypoints_ned(:,i)=R_e_n_Origin*deltaECEF;
    end
end

pitch_ref=0.0;
yaw_ref=current_yaw_rad;
Buoy_ref=0.0;
mission_complete=0;
target_xy_ned=[0.0;0.0];
debug_outputs=[Dist_to_Target;D_norm_filtered(1);D_norm_filtered(2)];
yaw_roll_coupling_gain=0.0;

if current_wp_index<=num_waypoints
    target_ned_3d=waypoints_ned(:,current_wp_index);
    target_xy_ned=[target_ned_3d(1);target_ned_3d(2)];
else
    target_ned_3d=waypoints_ned(:,num_waypoints);
    target_xy_ned=[target_ned_3d(1);target_ned_3d(2)];
end

switch glider_state

    case STATE_FIX_PLAN
        pitch_ref=pitch_antenna_rad;
        Buoy_ref=buoy_up_ref;
        yaw_ref=current_yaw_ref;
        yaw_roll_coupling_gain=0.0;

        time_in_state=t-t_state_start;

        if time_in_state>=surface_wait_time_s

            lon_glider_rad=current_lla_fix(1)*(pi/180);
            lat_glider_rad=current_lla_fix(2)*(pi/180);

            lla_glider_h0_rad=[lon_glider_rad;lat_glider_rad;0.0];
            ecef_glider_h0=fcn_lla2ecef(lla_glider_h0_rad);

            deltaECEF_glider=ecef_glider_h0-ECEF_Origin;

            ned_glider_3d=R_e_n_Origin*deltaECEF_glider;
            current_n=ned_glider_3d(1);
            current_e=ned_glider_3d(2);

            if max_depth_this_dive>min_estimation_depth_m
                current_pos_ned=[current_n;current_e];

                Expected_Glide_Vector=(2*max_depth_this_dive*baseline_glide_ratio)...
                                     *[cos(yaw_ref_last_dive);sin(yaw_ref_last_dive)];
                Expected_Pos_ned=dive_start_ned+Expected_Glide_Vector;

                Disturbance_Vector=current_pos_ned-Expected_Pos_ned;

                D_norm_estimated=Disturbance_Vector/max_depth_this_dive;

                D_norm_filtered=(alpha_filter*D_norm_estimated)+...
                                ((1-alpha_filter)*D_norm_filtered);
            end

            target_n=target_xy_ned(1);
            target_e=target_xy_ned(2);

            error_n=target_n-current_n;
            error_e=target_e-current_e;
            dist_to_wp=sqrt(error_n^2+error_e^2);
            dist_to_wp_last_fix=dist_to_wp;

            if dist_to_wp<acceptance_radius_m
                current_wp_index=current_wp_index+1;

                if current_wp_index>num_waypoints
                    glider_state=STATE_DONE;
                    return;
                end

                target_ned_3d=waypoints_ned(:,current_wp_index);
                target_xy_ned=[target_ned_3d(1);target_ned_3d(2)];

                error_n=target_xy_ned(1)-current_n;
                error_e=target_xy_ned(2)-current_e;
                dist_to_wp=sqrt(error_n^2+error_e^2);
                dist_to_wp_last_fix=dist_to_wp;
            end

            current_pos_ned=[current_n;current_e];
            Vector_to_Real_Target=target_xy_ned-current_pos_ned;
            Dist_to_Target=norm(Vector_to_Real_Target);

            horizontal_baseline=2*max_dive_depth_m*baseline_glide_ratio;

            Depth_guess_single=max_dive_depth_m;
            Single_Dive_Drift_Vector=D_norm_filtered*Depth_guess_single;

            Num_Dives_Needed=Dist_to_Target/(horizontal_baseline+1e-6);

            Total_Predicted_Drift=Single_Dive_Drift_Vector*Num_Dives_Needed;

            Required_Glide_Vector=Vector_to_Real_Target-Total_Predicted_Drift;
            Required_Horizontal_Dist=norm(Required_Glide_Vector);

            debug_outputs=[Dist_to_Target;D_norm_filtered(1);D_norm_filtered(2)];

            current_yaw_ref=atan2(Required_Glide_Vector(2),Required_Glide_Vector(1));

            yaw_ref_last_dive=current_yaw_ref;

            current_dive_depth_m=Required_Horizontal_Dist/(2*baseline_glide_ratio);

            current_dive_depth_m=max(current_dive_depth_m,initial_turn_depth_m*2.0);
            current_dive_depth_m=min(current_dive_depth_m,max_dive_depth_m);

            dive_start_ned=[current_n;current_e];
            max_depth_this_dive=0;
            glider_state=STATE_SUBMERGE;
            t_state_start=t;
        end

    case STATE_SUBMERGE
        pitch_ref=pitch_down_rad;
        Buoy_ref=buoy_down_ref;
        yaw_ref=current_yaw_rad;
        max_depth_this_dive=max(max_depth_this_dive,depth_sensor_m);
        if depth_sensor_m>initial_turn_depth_m
            glider_state=STATE_DIVE_TURN;
            t_state_start=t;
        end

    case STATE_DIVE_TURN
        pitch_ref=pitch_down_rad;
        Buoy_ref=buoy_down_ref;
        diff=current_yaw_ref-current_yaw_rad;
        wrapped_diff=atan2(sin(diff),cos(diff));
        best_yaw_ref=current_yaw_rad+wrapped_diff;
        current_yaw_ref=best_yaw_ref;
        yaw_ref=current_yaw_ref;
        yaw_roll_coupling_gain=1.0;
        max_depth_this_dive=max(max_depth_this_dive,depth_sensor_m);
        if (current_dive_depth_m-depth_sensor_m)<depth_epsilon_m
            glider_state=STATE_BOTTOM_WAIT;
            t_state_start=t;
        end

    case STATE_BOTTOM_WAIT
        pitch_ref=0.0;
        Buoy_ref=0.0;
        yaw_ref=current_yaw_ref;
        max_depth_this_dive=max(max_depth_this_dive,depth_sensor_m);
        time_in_state=t-t_state_start;
        if time_in_state>bottom_wait_time_s
            % Reset flare state for the new climb
            flare_active=false;
            glider_state=STATE_CLIMB;
            t_state_start=t;
        end

    case STATE_CLIMB
        Buoy_ref=buoy_up_ref;
        yaw_ref=current_yaw_ref;
        yaw_roll_coupling_gain=-1.0;

        if ~flare_active
            pitch_ref=pitch_up_rad;
            if depth_sensor_m<=flare_depth_m
                flare_active=true;
                t_flare_start=t;
                pitch_at_flare_entry=pitch_up_rad;
            end
        else
            flare_elapsed=t-t_flare_start;
            flare_progress=flare_elapsed/flare_duration_s;
            flare_progress=max(0.0,min(1.0,flare_progress));
            pitch_ref=pitch_at_flare_entry+(pitch_flare_target-pitch_at_flare_entry)*flare_progress;
        end

        if depth_sensor_m<=surface_epsilon_m
            flare_active=false;
            glider_state=STATE_SURFACE_WAIT;
            t_state_start=t;
        end

    case STATE_SURFACE_WAIT
        pitch_ref=pitch_antenna_rad;
        Buoy_ref=buoy_up_ref;
        yaw_ref=current_yaw_ref;

        time_in_state=t-t_state_start;
        if time_in_state>=surface_stabilize_time_s
            glider_state=STATE_FIX_PLAN;
            t_state_start=t;
        end

    case STATE_DONE
        pitch_ref=pitch_antenna_rad;
        mission_complete=1;

end

end

function ecef_pos=fcn_lla2ecef(lla_rad_vec)
    r_e=6378137.0;
    e_sq=0.00669437999014;
    lon_rad=lla_rad_vec(1);
    lat_rad=lla_rad_vec(2);
    h=lla_rad_vec(3);
    sin_lat=sin(lat_rad);
    cos_lat=cos(lat_rad);
    sin_lon=sin(lon_rad);
    cos_lon=cos(lon_rad);
    N=r_e/sqrt(1.0-e_sq*sin_lat^2);
    x=(N+h)*cos_lat*cos_lon;
    y=(N+h)*cos_lat*cos_lon;
    z=(N*(1.0-e_sq)+h)*sin_lat;
    ecef_pos=[x;y;z];
end

function R_n_e=fcn_get_R_n_e_matrix(l_rad,mu_rad)
    c_l=cos(l_rad);
    s_l=sin(l_rad);
    c_mu=cos(mu_rad);
    s_mu=sin(mu_rad);
    R_n_e=[-c_l*s_mu,-s_l,-c_l*c_mu;
           -s_l*s_mu, c_l,-s_l*c_mu;
            c_mu,      0,  -s_mu];
end