%% plot phase shifted signals

h=figure('Position',[1 1 1400 400]);
   
scale1 = 1;                     % y scaline of signal 1
scale2 = 3;                     % y scaline of signal 2    
scale3 = 5;                     % y scaline of signal 3
t = [0:0.05:10]';               % define time vector
 
phase_shift = [0 0.5 2];       % define phase shift vector

for count = 1:length(phase_shift)
    shift  = phase_shift(count);
    % generate phase shifted data
    y1 = scale1*sin(2*pi*t)+randn([length(t),1])/(100);
    y2 = scale2*sin(2*pi*t+0.25*shift*2*pi)+randn([length(t),1])/(100);
    y3 = scale3*sin(2*pi*t+shift*2*pi)+randn([length(t),1])/(100);
    y = [y1,y2,y3];
    
    subplot(1,3,count)
    plot(t,y);
    axis tight;
    xlabel('time [s]');
    ylabel('y');
    xlim([0 2]);
    
    if count == 1
        legend('task S1','task S2','task S3');
        title('Phase shift phi = 0');
    elseif count == 2
        title('Phase shift phi = \pi');
    elseif count ==3
        title('Phase shift phi = 4\pi');
    end
end