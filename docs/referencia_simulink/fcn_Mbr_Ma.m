function [M , m , I_b , r_g, MA, CA, CA_T, IA]  = fcn(r_lin,gamma,r_p,dist_xg,dist_yg)

m_s=37; % static mass
m_lin=15;    % Linear  mass
m_rot=5; %Rotational  mass
m_p=1; %piston  mass

m=m_s+m_lin+m_rot+m_p;
% Position vectors
r_s=[-0.1241; 0; 0.001];  % Hull CoM
R_r=0.07;  % m
r_rot=[0.2; R_r*cos(gamma+pi/2); R_r*sin(gamma+pi/2)]; %rotating mass position

I_s=[0.9 0 0; 0 21 0; 0 0 21]; %static mass inertia


I_lin=[0.02 0 0;   % Linear Mass Inertia
           0 10.16 0;
           0 0 0.17];


 I_rot=[0.00241296, 0, 0; %Rotating Mass Inertia
           0, 0.00207504, 0;
           0, 0, 0.00422813];


I_p=[0.00045521678, 0,  0;
              0,       0.01245908523,       0;
              0,            0,  0.01245908523];

I3=eye(3);

Rot_x=[1, 0, 0;
         0, cos(gamma), -sin(gamma);
         0, sin(gamma), cos(gamma)];

I_rot_rotated=1*Rot_x'*I_rot*Rot_x; %rotated mass inertia


% MA   = [3.9112 0 0; 0 58.1700 0; 0 0 48.4330]*1;
% IA   = [0.5 0 0; 0 15.83 0; 0 0 15.83]*1;
 CA   = [0 0 0; 0 0 0; 0 2.57*0 0]*1;
% CA_T = [0 0 0; 0 0 0; 0 3.61*0 0]*1;

MA   = [3.9112    0.1         0.1;
       0.1      58.1700       0.1;
       0.1          0.1      48.4330]*1;
IA   = [0.5 0.01  0.01; 0.01 15.83 0.01; 0.01 0.01 15.83]*1;
%CA   = [0 0 0; 0 0 0; 0 0 0]*1;
%CA_T = [0 0 0; 0 0 0; 0 3.61*0 0]*1;
CA_T=CA';
M_A = [MA, CA;
       CA_T, IA];

%M_RB2= [(m_s+m_lin+m_rot)*I3                        -m_s*s(r_s)-m_lin*s(r_lin)-m_rot*s(r_rot);
%        m_s*s(r_s)+m_lin*s(r_lin)+m_rot*s(r_rot)        I_s+I_rot_rotated+I_lin-m_s*s(r_s)*s(r_s)-m_rot*s(r_rot)*s(r_rot)-m_lin*s(r_lin)*s(r_lin)  ];

%M=M_RB2+M_A;
M_RB2=[ m*I3                                                           -m_s*s(r_s)-m_lin*s(r_lin)-m_rot*s(r_rot)-m_p*s(r_p);
         m_s*s(r_s)+m_lin*s(r_lin)+m_rot*s(r_rot)+m_p*s(r_p)            I_s+I_rot_rotated+I_lin+I_p-m_s*s(r_s)*s(r_s)-m_rot*s(r_rot)*s(r_rot)-m_lin*s(r_lin)*s(r_lin)-m_p*s(r_p)*s(r_p) ];
M=M_RB2 + M_A;

%r_g = (m_s*r_s + m_lin*r_lin + m_rot*r_rot) / m;
r_g=(m_s*r_s + m_lin*r_lin + m_rot*r_rot + m_p*r_p) / m;
r_g(1)=r_g(1)+dist_xg;
r_g(2)=r_g(2)+dist_yg;

I_b=I_s+I_rot_rotated+I_lin+I_p-m_s*s(r_s)*s(r_s)-m_rot*s(r_rot)*s(r_rot)-m_lin*s(r_lin)*s(r_lin)-m_p*s(r_p)*s(r_p);
end

function Skew = s(v)
     Skew = [    0   -v(3)   v(2);
           v(3)      0   -v(1);
          -v(2)   v(1)      0];
 end
