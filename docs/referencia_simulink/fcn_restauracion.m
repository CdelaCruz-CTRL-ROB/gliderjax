function T= fcn(W,B,r_g,Q,x_b)

phi=Q(4);
theta=Q(5);
psi=Q(6);
cphi=cos(phi);
sphi=sin(phi);
cthe=cos(theta);
sthe=sin(theta);
cpsi=cos(psi);
spsi=sin(psi);
%F_S_inertial = [0; 0; SURFACE_F]; %Force in z direction

R_b_n=[cthe*cpsi, sphi*sthe*cpsi - cphi*spsi, cphi*sthe*cpsi + sphi*spsi;
         cthe*spsi, cphi*cpsi + sphi*sthe*spsi, -sphi*cpsi + cphi*sthe*spsi;
         -sthe,     sphi*cthe,                  cphi*cthe];

%F_S_body = R_b_n' * F_S_inertial; %R_b_n'=R_n_b



x_g=r_g(1);
y_g=r_g(2);
z_g=r_g(3);
%x_b=0;
y_b=0;
z_b=0;
T=[(W-B)*sin(Q(5));
   -(W-B)*cos(Q(5))*sin(Q(4)) ;
   -(W-B)*cos(Q(4))*cos(Q(5));
    -(y_g*W-y_b*B)*cos(Q(4))*cos(Q(5))+(z_g*W-z_b*B)*sin(Q(4))*cos(Q(5));
    +(z_g*W-z_b*B)*sin(Q(5))+(x_g*W-x_b*B)*cos(Q(4))*cos(Q(5));
    -(x_g*W-x_b*B)*sin(Q(4))*cos(Q(5))-(y_g*W-y_b*B)*sin(Q(5))  ;];
