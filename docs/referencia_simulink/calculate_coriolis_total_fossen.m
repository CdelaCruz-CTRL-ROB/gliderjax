function [C_total, C_RB,C_A,testCa,testCrb, teststab] = calculate_coriolis_total_fossen(m, Ib, rg_b, MA, CA, CA_T, IA, v)


    v1 = v(1:3); % [u; v; w]
    v2 = v(4:6); % [p; q; r]

   S = @(a) [ 0 , -a(3), a(2);
               a(3), 0 , -a(1);
              -a(2), a(1), 0 ];

    % C_RB
    S_v2 = S(v2);
    S_rg_b = S(rg_b);

    C_RB = zeros(6, 6);
    C_RB(1:3, 1:3) = m * S_v2;
    C_RB(1:3, 4:6) = -m * S_v2 * S_rg_b;
    C_RB(4:6, 1:3) = m * S_rg_b * S_v2;
    C_RB(4:6, 4:6) = -S(Ib * v2);

    % C_A
    a1 = MA * v1 + CA * v2;
    a2 = CA_T * v1 + IA * v2;
    a3=MA*v2;

    S_a1 = S(a1);
    S_a2 = S(a2);
    S_a3=S(a3);


    C_A = zeros(6, 6);
    C_A(1:3, 1:3) = zeros(3, 3);
    %C_A(1:3, 1:3) = S_a3;
    C_A(1:3, 4:6) =  -S_a1;
    C_A(4:6, 1:3) =  -S_a1;
    C_A(4:6, 4:6) = -S_a2;
   %  C_A= [        0,               0,                 0,                       0,            48.433*v1(3),           -58.17*v1(2);
   %                0,               0,                 0,                -48.433*v1(3),            0,                 3.9112*v1(1);
    %               0,               0,                 0,                    58.17*v1(2),      -3.9112*v1(1),             0;
    %               0,          48.433*v1(3),    -58.17*v1(2),                   0,           15.83*v2(3),            -15.83*v2(2);
    %           -48.433*v1(3),         0,         3.9112*v1(1),             -15.83*v2(3),          0,                     0.5*v2(1);
     %           58.17*v1(2),    -3.9112*v1(1),        0,                    15.83*v2(2),      -0.5*v2(1),                       0];

    teststab=[v1; v2]'*C_A*[v1; v2];
    testCa=C_A+C_A';
    testCrb=C_RB+C_RB';
    %Total Coriolis Matrix
    C_total = C_RB+C_A*0 ;

end
