
clear;
close all;
clc;
load Measurements.mat
load DEM_heights.mat
load DEM_Complete.mat 
load DataV4.mat
load ProcNoise2.mat

% Reference:LongZhao 2015
% Gross 2010

% Load the dataset/measurement

% load Data_Karachi.mat
% 
% x_gps = DataV1(2:2001,8);
% y_gps = DataV1(2:2001,9);
% 
% figure,
% subplot(211)
% plot(x_gps,'r');grid on,
% subplot(212)
% plot(y_gps,'g');grid on,

% Add the noise to the measurement

% load TERCOMPos.mat

% load ProcNoise.mat
load ProcNoise2.mat

h_db = h_baro(700:750)-h_radar(700:750);
figure,
plot(h_db,'r');grid on,
figure,
hist(h_db)% repetitive terrain



% Initialize the PF

%% NumberOfParticles : N = 200.

N =  2000;

% sample particles/initialize using the normal
% distribution

particles = randn(2,N);  % row one for x-position and row two for y-position.

temp = particles;

iteration = 1;

Pk = [1e-9 1e-8;
      1e-6 2e-9];
  
  c = cell(N,1);
  
for i = 1:N
    c{i,1}=Pk;
end


  
Qk = [0.1 0;
      0    0.1];

Rk = [0.04 0
      0   0.01];  % decreasing R is good

  
tic  
for k = 1:50 % time steps

% record the sensor measurement
% we assume the measurement is from the GPS

 m = abs(Z-mean(mean(abs(Z-h_db(k)))));
    [r,cl]=find(m==min(min(m)));
    row=r(1);
    col=cl(1);
    pos=M{row,col};
    Lat_tercom(k)=pos(1);
    Long_tercom(k)=pos(2);
   k;
    z = [Lat_tercom(k) Long_tercom(k)]';

    
for i = 1:N  % repeat for each particle


% The Prediction Step
% ===================

Xpred(1:2,i)=particles(1:2,i);

% Wk = randn(2,1);  % process noise vector for the prediction phase
Wk = P2(k).*randn(2,1);

% Make the Prediction:
% ===================

% Define the Jacobian / Process Model:

F = [1 0;
     0 1];
% X = [x
%     y];

Xpred(1:2,i) = F*Xpred(1:2,i)+Wk;   % predict state 


Pk = c{i,1};
Pk = F*Pk*F'+Qk;      % predict covariance
c{i,1}=Pk;

% Measurement Update / Correction
% ===============================

%  At the availability of a GPS measurement, the EKF update procedure is performed
%  on each particle.

% Compute the Innovation

h = [1 0;
     0 1]; % observation matrix

I(1:2,i) = z-h*Xpred(:,i);  % (2 x 200) matrix = number of states*number of particles
                               %  I is sometimes denoted by 'y'.

% Compute the Matrix 'S'

S = h*Pk*h'+Rk;

% Compute the Kalman Gain

K = Pk*(h')*inv(S);

% update the State

Xupdt(:,i) = Xpred(:,i)+K*I(1:2,i);

% Pk = (I(1:2,i)-K*h)*Pk;
Pk = (eye(2)-K*h)*Pk;

c{i,1}=Pk;

% update the particles

particles(:,i)= Xupdt(:,i);

end

% Now we have got the updated particles

% Arupalam Paper:
% Now assign each particle a weight

if k==1
    % Measurement Likelihood
dist =sqrt((Xupdt(1,:)-z(1)).^2+(Xupdt(2,:)-z(2)).^2);
w=1./dist;  % Assign each particle a weight
% Normalize each weight
w = w./sum(w);
end

% Test whether resampling is required
% Compute Neff:
Neff(k) = 1./sum(w.^2);

% Specify Threshold
Thresh = 300;

if Neff(k)<Thresh
    % Apply resampling
    cdf = 0;
    cdf=cumsum(w);
    i=1;
    u1 = ((1/N)-0)*unifrnd(1,1);
    % move along the CDF
    for j = 1:N-1
    uj = u1+((j-1)/N);
    while uj>cdf(i)
        i = i+1;
    end
    Xupdt(1:2,j)=Xupdt(1:2,i);
    w = ones*(1/N);
%     i(j)=i;
    end
% Xcorr(1:2,k)=mean(Xupdt,2);   
else
% Xcorr(1:2,k)=mean(Xupdt,2);
end

% Xcorr(1:2,k)=mode(Xupdt,2);

Xcorr(1:2,k)=sum(Xupdt.*w,2);

disp(k)
end

time=toc 

xgps= DataV4(700:750,8);
ygps= DataV4(700:750,9);

rmse_x = sqrt(mean((ygps(1:50)-Lat_tercom(1:50)').^2))
rmse_y= sqrt(mean(xgps(1:50)-Long_tercom(1:50)').^2)

rmse_xpf = sqrt(mean((ygps(1:50)-Xcorr(1,1:50)').^2))
rmse_ypf = sqrt(mean((xgps(1:50)-Xcorr(2,1:50)').^2))







