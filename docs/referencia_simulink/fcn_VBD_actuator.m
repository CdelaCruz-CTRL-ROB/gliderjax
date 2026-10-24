

function [r_p, x_b, B_total, percent_fill, p_ref, delta_x,delta_V_actual] = fcn_VBD_actuator(delta_B_ref, con)
    rho=1000;      % Density
    g=9.81;        % Gravity

    W_total=58*9.81; % Weight
    B_neutral=W_total;


    rb_neutral=[0;0;0];

    diam=0.08;           % 80 mm
    area=pi*(diam/2)^2;

    stroke_total = 0.200;  % 200 mm
    max_disp=stroke_total/2;

    x_0_vbd=-0.7;
    y_p=0;
    z_p=0;

    delta_V_req=delta_B_ref/(rho*g);
    delta_x_req=-delta_V_req/area;

    if delta_x_req>max_disp
        delta_x=max_disp;
    elseif delta_x_req<-max_disp
        delta_x=-max_disp;
    else
        delta_x=delta_x_req;
    end


    %  Piston Position
    r_p=[x_0_vbd+delta_x;
           y_p;
           z_p];

    % Buoyancy
    delta_V_actual = -delta_x * area;
    B_total = B_neutral + (delta_V_actual * rho * g);

    % Buoyancy center (r_b)
     x_centroid_delta=x_0_vbd+(delta_x/2);

    delta_B_actual=delta_V_actual*rho*g;

    if B_total>0
        xb_new=(B_neutral*rb_neutral(1)+delta_B_actual*x_centroid_delta)/B_total;
    else
        xb_new=0;
    end

    r_b=[xb_new;
         rb_neutral(2);
         rb_neutral(3)];
         x_b=xb_new;

    percent_fill=(delta_x/max_disp)*100;

    if delta_B_ref<0
        p_ref=-con;
    elseif delta_B_ref==0
        p_ref=0;
    else
        p_ref=con;
    end
end
