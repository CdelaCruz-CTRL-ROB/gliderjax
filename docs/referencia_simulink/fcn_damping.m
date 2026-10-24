% REFERENCIA: función de amortiguamiento del modelo Simulink original
% (body damping + flow damping), copiada tal cual la proporcionó el autor
% el 2026-10-24. Es la "verdad" contra la que se valida
% gliderjax/dynamics.py::tau_flowframe y tau_rot_damping.
% Ver docs/MODELO.md, sección 2.

function [D_total,  alpha,beta,Vres] = fcn(V, Q,  r_cw)
% body damping + flow damping
%   V:   [u; v; w; p; q; r]
%   Q:   [x; y; z; phi; theta; psi]

% BODY DAMPING
% D_B2 = [4.5*V(1)*V(1)*sign(V(1)) + 0.7*V(1);
%           %8*V(2)+280*V(2)*V(2)*sign(V(2))+3*V(3)*V(2)+14*V(2)*V(1)+18*V(6)*V(1);
%           12*V(2)+380*V(2)*V(2)*sign(V(2))+14*V(2)*V(1)+5*V(6)*V(1);
%           12*V(3)+400*V(3)*V(3)*sign(V(3))+270*V(3)*V(1);
%           20*V(4)+120*V(4)*V(4)*sign(V(4))-50*V(4)*V(2)   ;
%           2*V(5)+120*V(5)*V(5)*sign(V(5))+5*V(5)*V(1)-r_cw*250*V(3)*V(3)*sign(V(3))+35*V(5)*V(1);
%           200*V(6)*V(6)*sign(V(6))-3.6*120*V(2)*V(2)*sign(V(2))];


% D_B2 = [2*4.5*V(1)*V(1)*sign(V(1)) + 2*0.7*V(1);
%         12*V(2)+380*V(2)*V(2)*sign(V(2))+0*14*V(2)*V(1)+0*5*V(6)*V(1);
%         12*V(3)+400*V(3)*V(3)*sign(V(3))+0*270*V(3)*V(1);
%         20*V(4)+120*V(4)*V(4)*sign(V(4))-0*50*V(4)*V(2)   ;
%         2*V(5)+120*V(5)*V(5)*sign(V(5))+0*5*V(5)*V(1)-0*r_cw*250*V(3)*V(3)*sign(V(3))+0*35*V(5)*V(1);
%         200*V(6)*V(6)*sign(V(6))-0*3.6*120*V(2)*V(2)*sign(V(2))];

D_body = [0;
          0;
          0;
          0.2*20*V(4) + 0.2*120*V(4)*V(4)*sign(V(4))   ;
          0.5*20*V(5) + 0.5*100*V(5)*V(5)*sign(V(5));
          100*V(6)*V(6)*sign(V(6))];
% euler
phi   = Q(4);
theta = Q(5);
psi   = Q(6);

% Fllow frame coefficients
KD0    =5.9;
KD     = 214.62;
Kbeta  = -56.66 ;
KL0    = 0.326713*0;
Kalpha = 308.9754;
KMR    = 4.5194;
KM0    = -0.0328;
KM     = -12.48;

KMYLIN    = 15.04 ;
KMYCUB    =-398.442;

% Rotation matrix (body to inertial)
cphi = cos(phi);   sphi = sin(phi);
cthe = cos(theta); sthe = sin(theta);
cpsi = cos(psi);   spsi = sin(psi);

R_EB = [cthe*cpsi, sphi*sthe*cpsi - cphi*spsi, cphi*sthe*cpsi + sphi*spsi;
        cthe*spsi, cphi*cpsi + sphi*sthe*spsi, -sphi*cpsi + cphi*sthe*spsi;
        -sthe,     sphi*cthe,                  cphi*cthe];

Vr = V(1:3);
% Flow angles
Vres = sqrt(Vr(1)^2 + Vr(2)^2 + Vr(3)^2);
alpha = atan2(Vr(3), Vr(1));
KMR=KMR*sign(alpha);
if (Vres == 0)
    beta = 0.0;
else
    beta = asin(Vr(2) / Vres);
end

% Forces in flow frame
D  = (KD0 + KD*alpha^2) * Vres^2;
SF = Kbeta * beta * Vres^2;
L  = (KL0 + Kalpha*alpha) * Vres^2;

% Moments in flow frame
TDL1 = (KMR*beta) * Vres^2;
TDL2 = (KM0 + KM*alpha) * Vres^2;
TDL3 = (KMYLIN*beta+KMYCUB*beta^3) * Vres^2;

Fh = [-D; SF; -L];
Th = [TDL1; TDL2; TDL3];

% Rotation matrix (flow to body)
R_CB = [cos(alpha)*cos(beta), -cos(alpha)*sin(beta), -sin(alpha);
        sin(beta),             cos(beta),             0;
        sin(alpha)*cos(beta), -sin(alpha)*sin(beta),  cos(alpha)];

F = R_CB * Fh;
T = R_CB * Th;
D_flow = -[F; T];

Model_A = D_body + D_flow;

D_total=Model_A;
