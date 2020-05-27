channels = [8,16,19,24,38,64];
inter = [56.20,62.56,63.63,63.97,64.61,65.54];
intra = [57.62,63.44,65.54,65.77,66.82,67.41];

channels_ori = [8,19,38,64];
inter_ori = [60.27,63.39,64.15,65.54];
intra_ori = [61.26, 64.42,66.22,67.41];
% ----
acc_t = [63.32,64.99,65.90,65.54];
t = [0.5,1,2,3];
% ----
acc_shift = [63.32,62.63,58.30,50.83,43.57];
t_shift = [0.1,0.3,0.5,0.7,0.9];
% ----
class = [2,3,4];
inter_intra_cl = [82.09 83.72; 74.53 76.37; 65.54 67.19];
% ----

figure(1);
plot(channels, inter, '-x');
hold on;
plot(channels, intra, '-x');
plot(channels_ori, inter_ori, '-x');
plot(channels_ori, intra_ori, '-x');
xlabel('Number of Channels');
ylabel('Validation Accuracy (%)');
title('4 CLASS VALIDATION ACCURACY AGAINST NUMBER OF CHANNELS (T = 3s, ds = 1)');
legend('inter-subject', 'intra-subject', 'inter-subject original', 'intra-subject original');
grid on;
grid minor;
hold off;

figure(2);
plot(t, acc_t, '-x');
xlabel('T (s)');
ylabel('Accuracy (%)');
title('Accuracy vs Time Window Length');
grid on;
grid minor;

figure(3);
bar(t_shift, acc_shift);
xlabel('Starting Time (s)');
ylabel('Accuracy (%)');
title('ACCURACY VS STARTING TIME (T=1s, 64 CHANNELS, DS=1)');
grid on;
grid minor;

figure(4);
bar(class, inter_intra_cl);
ylim([60 85]);
xlabel('Number of Classes');
ylabel('Accuracy (%)');
title('VALIDATION ACCURACY FOR MOTOR MOVEMENT CLASSIFICATION (DS=1, 64CH, T=3s)');
legend('inter-subject', 'intra-subject');
grid on;
grid minor;