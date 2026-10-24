function[B_scale,delta_xb]=fcn_buoyancy_surface(z,pitch_rad)

L=1.8;
Lh=L*0.5;
H=0.2;
Hh=H*0.5;

sp=sin(pitch_rad);
cp=cos(pitch_rad);

z_nose_top=z-Lh*sp-Hh*cp;
z_tail_top=z+Lh*sp-Hh*cp;

emg_nose=max(0.0,-z_nose_top)/H;
emg_nose=min(emg_nose,1.0);

emg_tail=max(0.0,-z_tail_top)/H;
emg_tail=min(emg_tail,1.0);

avg_emg=(emg_nose+emg_tail)*0.5;

B_scale=1.0-avg_emg;

emg_sum=emg_nose+emg_tail;
if emg_sum>1e-8
    delta_xb=-Lh*(emg_nose-emg_tail)/emg_sum* avg_emg;
else
    delta_xb=0.0;
end

end
