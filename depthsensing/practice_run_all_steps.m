clear all; close all; clc;
format long;

%==========================================================================
%geometric setup of the three cameras
%==========================================================================
% check data/elephant/info_dict.json
back_dist = 16;
baseline = 16;
focal_length = 7200;
chess_block = 8.433;
%back_dist = 20.577952178726964;
%baseline = 20.577952178726964;      
%focal_length = 43962.93892852579;   % pixels
%image_dir = './data/elephant';

cx = -31.146606;cy = 11.0932;
image_dir = 'E:/depsensing/data/midtrans7';
disp_dir = 'E:/depsensing/output7/disp_esti';
out_dir = 'E:/depsensing/output7';
CUDA_DEVICE = '0';
filename = 'E:/depsensing/chessdata.txt'; 
strchesspoint= dlmread(filename, '\t'); 

% cx = -33.9570313;cy = 10.2587385;
% image_dir = 'E:/depsensing/data/midstr';
% disp_dir = 'E:/depsensing/output3window/disp_esti';
% out_dir = 'E:/depsensing/output3window';
% CUDA_DEVICE = '0';
% filename = 'E:/depsensing/strdata.txt'; 
% strchesspoint= dlmread(filename, '\t'); 

% cx = -33.9570313;cy = 10.2587385;
% image_dir = 'E:/depsensing/data/midbuild';
% disp_dir = 'E:/depsensing/output5building/disp_esti';
% out_dir = 'E:/depsensing/output5building';
% CUDA_DEVICE = '0';
% filename = 'E:/depsensing/builddata.txt'; 
% strchesspoint= dlmread(filename, '\t'); 

if ~exist(out_dir, 'dir')
   mkdir(out_dir)
end
%==========================================================================
%==========================================================================
% pseudo-rectify
%==========================================================================
%==========================================================================
color_I1 = imread([image_dir, '/color_left.png']);
color_I2 = imread([image_dir, '/color_right.png']);
%==========================================================================
% match surf key points
%==========================================================================
I1 = rgb2gray(color_I1); I2 = rgb2gray(color_I2);

%matchedPoints1 = feature1(:,matchesij(1, :));
%matchedPoints2 = feature2(:,matchesij(2, :));
%showMatchesSIFT(color_I1,color_I2,matchedPoints1,matchedPoints2);

points1 = detectSURFFeatures(I1,'MetricThreshold',2000);
points2 = detectSURFFeatures(I2,'MetricThreshold',2000);
[f1,vpts1] = extractFeatures(I1,points1);
[f2,vpts2] = extractFeatures(I2,points2);

indexPairs = matchFeatures(f1,f2,'MatchThreshold',2.0) ;
left_matched_points = vpts1(indexPairs(:,1));
right_matched_points = vpts2(indexPairs(:,2));

% visualize the matches for debugging purposes
figure;
subplot(121);
imshow(I1); hold on;
plot(left_matched_points.selectStrongest(100));
xlabel('Left View');
subplot(122);
imshow(I2); hold on;
plot(right_matched_points.selectStrongest(100));
xlabel('Right View');
% % sgtitle('Strongest 100 Surf Matches');


delta_y = abs(left_matched_points.Location(:, 2) - right_matched_points.Location(:, 2));
valid_idx = delta_y <= 100;
left_matched_points = left_matched_points(valid_idx, :);
right_matched_points = right_matched_points(valid_idx, :);

%left_matched_points = left_matched_points(valid_idx, :);
%right_matched_points = right_matched_points(valid_idx, :);
showMatchesSIFT(color_I1,color_I2,left_matched_points.Location',...
    right_matched_points.Location');
left_matched_points = left_matched_points.Location;
right_matched_points = right_matched_points.Location;
fprintf('Matched %i points\n', size(left_matched_points, 1));

%==========================================================================
% estimate 2*3 affine matrices that pseudo-rectify the left, right images
%==========================================================================

scale = 2000;
num_ransc_trials = 5000;
min_set_size = 10;
max_support = 0;
thres = 2;          % error below 2 pixels to be considered as inlier
x_diff = 0;         % the content of pseduo-rectified right image should move left-ward w.r.t left image
for i=1:num_ransc_trials
    % randomly sample the minimum set
    tmp = randperm(size(right_matched_points, 1));
    tmp = tmp(1:min_set_size);
    A = [left_matched_points(tmp, :) / scale, -right_matched_points(tmp, :) / scale, ones(min_set_size, 1)]; % use noisy matches
    
    [U,D,V] = svd(A,0);
    x1 = V(:,end);
    
    % make sure unit norm for the first two components
    x1(1:4) = x1(1:4) / scale;  % numerical trick
    x1 = x1 / norm(x1(1:2));
    % positive sign for a22
    x1 = x1 / sign(x1(2));
    
    % check size of the support set
    tmp = [left_matched_points, -right_matched_points, ones(size(left_matched_points, 1), 1)] * x1;
    mask = abs(tmp) < thres;
    support = sum(mask) / size(tmp, 1);
    %fprintf('ransac trial %i, support %.4f\n', i, support);
    if support > max_support
       max_support = support;
       x = x1;
       inlier_mask = mask;
    end
end
fprintf('End of ransac, max_support %.4f\n', max_support);

%==========================================================================
% compose affine matrices for both views
%==========================================================================
% left view
col_vec1 = x(1:2, :);
col_vec2 = [-col_vec1(2);col_vec1(1)];
rot_mat = [col_vec2, col_vec1];
% check determinant
if (det(rot_mat) < 0)
   col_vec2 = -col_vec2;
   rot_mat = [col_vec2, col_vec1];
end
affine_mat_1 = [rot_mat', [0., 0.]'];

% right view
col_vec1 = x(3:4, :);
col_vec2 = [-col_vec1(2);col_vec1(1)];
rot_mat = [col_vec2, col_vec1];
% check determinant
if (det(rot_mat) < 0)
   col_vec2 = -col_vec2;
   rot_mat = [col_vec2, col_vec1];
end
tmp = 0. - x(5, 1);
affine_mat_2 = [rot_mat', [0., tmp]'];

cnt = sum(inlier_mask(:));
x_diff = [right_matched_points(inlier_mask, :), ones(cnt, 1)] * reshape(affine_mat_2(1, :), 3, 1) ...
            - [left_matched_points(inlier_mask, :), ones(cnt, 1)] * reshape(affine_mat_1(1, :), 3, 1);
x_diff = median(x_diff);
margin = 50.0;
x_translation = -(x_diff + margin);
affine_mat_2(1, 3) = x_translation;

disp('Estimated affine matrix for left view:')
disp(affine_mat_1);
disp('Estimated affine matrix for right view:')
disp(affine_mat_2);

%==========================================================================
%apply pseudo-rectification and write rectified pairs
%==========================================================================
pseudo_rectify_dir = [out_dir, '/pseudo_rectify'];
if ~exist(pseudo_rectify_dir, 'dir')
   mkdir(pseudo_rectify_dir)
end

tform = affine2d([affine_mat_1', [0; 0; 1]]);
pseudo_rect_I1 = imwarp_same(color_I1, tform);

tform = affine2d([affine_mat_2', [0; 0; 1]]);
pseudo_rect_I2 = imwarp_same(color_I2, tform);
if 1
    csvwrite([pseudo_rectify_dir, '/affine_mat_im0.txt'], affine_mat_1);
	imwrite(color_I1, [pseudo_rectify_dir, '/orig_im0.png']);
	imwrite(color_I2, [pseudo_rectify_dir, '/orig_im1.png']);
	imwrite(pseudo_rect_I1, [pseudo_rectify_dir, '/im0.png']);
	imwrite(pseudo_rect_I2, [pseudo_rectify_dir, '/im1.png']);

%==========================================================================
% create a small area to visually inspect the quality of rectification
%==========================================================================
%figure;
%subplot(121);
%imshow(pseudo_rect_I1(1822:1922, 1871:1971, :));
%title('Crop of Rectified Left view');
%subplot(122);
%imshow(pseudo_rect_I2(1822:1922, 1851:1951, :));
%title('Crop of Rectified Right view');
%set(gcf,'color','w');

%==========================================================================
%==========================================================================
% run stereo matching
%==========================================================================
%==========================================================================
tmp_dir = [pseudo_rectify_dir, '/tmp'];
if ~exist(tmp_dir, 'dir')
   mkdir(tmp_dir)
end
%cmd = ['move ' pseudo_rectify_dir '\im0.png ' tmp_dir];
%status = system(cmd);
%disp(pseudo_rectify_dir)
%cmd = ['move ' pseudo_rectify_dir '\im1.png ' tmp_dir];
%status = system(cmd);

disp_esti_dir = [out_dir, '/disp_esti'];
disp(disp_esti_dir)
if ~exist(disp_esti_dir, 'dir')
   mkdir(disp_esti_dir)
end
% cmd = ['CUDA_VISIBLE_DEVICES=' CUDA_DEVICE ...
%       ' python3 high-res-stereo/submission.py ' ...
%       ' --datapath ' pseudo_rectify_dir ...
%       ' --outdir ' disp_esti_dir ...
%       ' --loadmodel high-res-stereo/final-768px.pth ' ...
%       ' --testres 0.5 --clean 1.0 --max_disp 512 '];
% 移动文件
cmd = ['move ' disp_esti_dir '\tmp\*.* ' disp_esti_dir];
status = system(cmd);
disp(cmd);
% 删除临时目录
cmd = ['rmdir /s /q ' disp_esti_dir '\tmp'];
status = system(cmd);
disp(cmd);
% 删除临时文件
cmd = ['del /q ' tmp_dir '\*.*'];
status = system(cmd);
disp(cmd);
% 删除临时目录
cmd = ['rmdir /s /q ' tmp_dir];
status = system(cmd);
disp(cmd);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%==========================================================================
% ambiguity removal
%==========================================================================
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%==========================================================================
%load left and back images, and the predicted disparity map
%==========================================================================
ambi_remove_dir = [out_dir, '/ambiguity_remove'];
if ~exist(ambi_remove_dir, 'dir')
   mkdir(ambi_remove_dir)
end
end%document moveing
color_I1 = imread([pseudo_rectify_dir, '/im0.png']);
%color_I2 = imread([disp_dir,'/disp.jpg']);
%color_I2 = imread([image_dir, '/color_back.png']);
color_I2 = imread([pseudo_rectify_dir, '/im1.png']);
I1 = rgb2gray(color_I1); 
I2 = rgb2gray(color_I2);
disparity = hdf5read([disp_esti_dir, '/disp.h5'],'data');
disparity = disparity';   

invalid_mask = I1 < 1e-5;   % black pixels

%figure;
mask_tmp = invalid_mask | isnan(disparity);
disparity_tmp = disparity;
disparity_tmp(mask_tmp) = max(disparity(~mask_tmp));
imagesc(disparity_tmp,'AlphaData',~mask_tmp);
title('Estimated Disparity');
set(gcf,'color','w');
colorbar('southoutside');

%imwrite( ind2rgb(im2uint8(mat2gray(disparity_tmp)), parula(256)), [ambi_remove_dir, '/disp_esti.png'], 'png', 'Alpha', uint8(~mask_tmp) * 255);

%==========================================================================
% match surf key points between left and back images
%==========================================================================
points1 = detectSURFFeatures(I1,'MetricThreshold',2000);
points2 = detectSURFFeatures(I2,'MetricThreshold',2000);
[f1,vpts1] = extractFeatures(I1,points1);
[f2,vpts2] = extractFeatures(I2,points2);
indexPairs = matchFeatures(f1,f2,'MatchThreshold',2.0) ;
forward_matched_points = vpts1(indexPairs(:,1));
backward_matched_points = vpts2(indexPairs(:,2));
forward_matched_points = forward_matched_points.Location;
backward_matched_points = backward_matched_points.Location;
%==========================================================================
% now try to estimate the horizontal ambiguity of disparity
%==========================================================================
max_trials = 50;
ambiguity = zeros(1, max_trials);
idx = 1;
idelta = 1;
Value1 = zeros(1, max_trials);
Value2 = zeros(1, max_trials);
num = zeros(1, max_trials);
fileID = fopen('data.txt', 'a');
delta_d = zeros(1, max_trials);
if fileID == -1
    error('File cannot be opened');
end
showchesspoint(color_I1,strchesspoint)
while idx <= max_trials    
    ii = randi([1,54]);
    while 1
        jj = randi([1,54]);
        if jj ~= ii
            break;
        end
    end
    forward_ii_x = strchesspoint(ii,1);
    forward_ii_y = strchesspoint(ii,2);
    disp1 = disparity(forward_ii_y,forward_ii_x);
    forward_jj_x = strchesspoint(jj,1);
    forward_jj_y = strchesspoint(jj,2);
    disp2 = disparity(forward_jj_y,forward_jj_x);    
    left_test = [forward_ii_x;forward_ii_y];
    right_test = [forward_jj_x;forward_jj_y];
    close(gcf,'all');
    %showMatchesdist(color_I1,left_test,right_test);
    kx = (forward_ii_x - cx)/disp1 - (forward_jj_x - cx)/disp2;
    ky = (forward_ii_y - cy)/disp1 - (forward_jj_y - cy)/disp2;
    k = sqrt(kx^2+ky^2);    
    bx = ceil(ii/9)- ceil(jj/9);
    y1 = mod(ii, 9);
    y2 = mod(jj, 9);
    if y1 == 0
        y1= y1+9;
    end
    if y2 == 0
        y2= y2+9;
    end
    
    by = abs(y1-y2);
    b = sqrt(bx^2+by^2);
    delta_d(1,idx) = k*baseline / (chess_block*b);
    C1=sqrt((forward_ii_x-cx)^2+(forward_ii_y-cy)^2+focal_length^2);
    C2=sqrt((forward_jj_x-cx)^2+(forward_jj_y-cy)^2+focal_length^2);
    X = C2-C1;
    X = X / abs(X);
    %%%
    b = chess_block*(disp1+disp2)/baseline+X*(C2-C1);
    a = chess_block / baseline;
    c = chess_block/baseline*disp1*disp2+X*(disp1*C2-disp2*C1);
    XX = b*b-abs(4*a*c);
    syms x;
    answ = solve(abs(C1*baseline/(disp1+x)-C2*baseline/(disp2+x)) == chess_block, x);
    %%%C1*baseline/(disp1+x)-C2*b/(disp2+x)) == chess_block
    if isreal(answ(1))
        disp(idx);
        %delta_d(1,idx)=(-b+sqrt(b*b-4*a*c))/(2*a);        
        delta_d(1,idx) = answ(1);
        %delta_d(2,idx) = answ(2);
    end
    %delta_d(1, idx) = kx * baseline /( chess_block*1);
    %delta_d(2, idx) = ky * baseline /( chess_block*0);
    %fprintf("bx is %f, by is %f, d1 is %f\n",bx, by, delta_d(1,idx));
    idx = idx + 1;
    %disp(bx);disp(by);disp(delta_d(1,idx));disp(delta_d(2,idx))
end
% while idx <= max_trials
%     % sample two pixels
%     %disp(forward_matched_points);
%     ii = randi(size(forward_matched_points, 1));
%     while 1
%         jj = randi(size(forward_matched_points, 1));
%         if jj ~= ii
%             break;
%         end
%     end
%     % check their disparity
%     forward_ii_x = forward_matched_points(ii, 1);
%     forward_ii_y = forward_matched_points(ii, 2);
%     forward_jj_x = forward_matched_points(jj, 1);
%     forward_jj_y = forward_matched_points(jj, 2);
%     
%     %backward_ii_x = backward_matched_points(ii, 1);
%     %backward_ii_y = backward_matched_points(ii, 2);
%     %backward_jj_x = backward_matched_points(jj, 1);
%     %backward_jj_y = backward_matched_points(jj, 2);
%     
%     forward_ii_x = round(forward_ii_x);
%     forward_ii_y = round(forward_ii_y);
%     forward_jj_x = round(forward_jj_x);
%     forward_jj_y = round(forward_jj_y);
%     
%     %backward_ii_x = round(backward_ii_x);
%     %backward_ii_y = round(backward_ii_y );
%     %backward_jj_x = round(backward_jj_x);
%     %backward_jj_y = round(backward_jj_y);
%     
%     % check the mask
%     if invalid_mask(forward_ii_y, forward_ii_x) || invalid_mask(forward_jj_y, forward_jj_x)
%         continue;
%     end
%     
%     % check the disparity
%     disp1 = disparity(forward_ii_y,forward_ii_x);
%     disp2 = disparity(forward_jj_y, forward_jj_x);
%     thres = 5;
%     if(isnan(disp1) ||...
%        isnan(disp2) ||...
%        abs(disp1 - disp2) > thres)
%        continue;
%     end
%     % check pixel distance
%     forward_dist = sqrt((forward_ii_x - forward_jj_x).^2 + ...
%                         (forward_ii_y - forward_jj_y).^2);
%     %backward_dist = sqrt((backward_ii_x - backward_jj_x).^2 + ...
%     %                    (backward_ii_y - backward_jj_y).^2);
%     thres = 200;
%     %if (forward_dist < thres || forward_dist < backward_dist)
%     %    continue;
%     %end
%     
%     % compute expected disparity
%    % expected_disp = focal_length * baseline / back_dist * (forward_dist / backward_dist - 1);
%     %diff = expected_disp - (disp1 + disp2) / 2;
%     
%     clip_thres = 1000;
%     %if (diff < 0 || diff > clip_thres)
%     %   continue; 
%     %end
%     
%     %ambiguity(1, idx) = diff;
%     %num(1,idx) = forward_ii_x;
%     num(1, idx) = disp2;
%     %delta_d(1, idx) = abs(disp1 - disp2);
%     %Value1(1, idx) = color_I2(forward_ii_x, forward_ii_y, :);
%     %Value2(1, idx) = color_I2(forward_jj_x, forward_jj_y, :);
%     %fprintf('Trial %i, Ambiguity %f\n', idx, diff);
%     
%     idx = idx + 1;
% end
% 关闭文件
fclose(fileID);
%adjust = median(ambiguity);
variance = var(delta_d, 0);
fprintf("variance is %f\n",variance);
% 计算样本均值和标准差
meanValue = mean(delta_d);
%meanValue = median(delta_d);
stdDev = std(delta_d);
adjust = meanValue;
disp('adjust: ');disp(adjust);
n = length(delta_d);
zScore = 1.96;
CI = [meanValue - zScore * (stdDev / sqrt(n)), meanValue + zScore * (stdDev / sqrt(n))];
disp(['95% 置信区间: [', num2str(CI(1)), ', ', num2str(CI(2)), ']']);
figure;
histogram(delta_d(1));
title('直方图');
xlabel('值');
ylabel('频数');
%figure;
%[x, fval] = lsqlin(Value1', Value2', [], [], [], [], [], []);
%将多个图画到一个平面上
%plot(num, delta_d, 'o', num, num*x, 'r', num, num*x, '+')
%axis([0 5 0 5]);          
%绘图时x、y轴的上下限
%grid on              
%fprintf(fileID, '%f\n', ambiguity);


%fprintf('Median Ambiguity %f\n', adjust);
%figure;
%[x, fval] = lsqlin(num', delta_d', [], [], [], [], [], []);
%subplot(1, 2, 1);       
%将多个图画到一个平面上
%plot(num, delta_d, 'o', num, num*x, 'r', num, num*x, '+')
%axis([0 350 0 6]);          
%绘图时x、y轴的上下限
%grid on                 
%在画图的时候添加网格线。
%xx = linspace(0, 1);
%linspace(x1,x2,N)均分指令，其中x1、x2、N分别为起始值、终止值、元素个数。若默认N，默认点数为100。
%logspace(a, b, n)生成一个(1*n)数组，数据的第一个元素值为10^a，最后一个元素为10^b，n是总采样点数。
%subplot(1, 2, 2);
%plot(num, num*x, '+', xx, x*xx, 'r')
%axis([0 2 0 0.3])
%grid on

% figure;
% ambiguity(ambiguity > clip_thres) = clip_thres;
% ambiguity(ambiguity < -clip_thres) = -clip_thres;
% h = histogram(ambiguity);
% hold on; 
% line([adjust, adjust], [0, max(h.Values) + 100], 'Color', 'r', 'LineWidth', 2);
% ylim([0, max(h.Values) + 100]);
% title('Distribution of All Cached Ambiguity Estimates');
% set(gcf,'color','w');

%==========================================================================
% add estimated ambiguity to the predicted disparity
% and convert disparity to depth
%==========================================================================
disparity = disparity - adjust;
esti_depth = focal_length * baseline ./ disparity;

%==========================================================================
% visualize results
%==========================================================================
figure;
nan_mask = isnan(esti_depth) | invalid_mask;

mask_tmp = nan_mask;
esti_depth_tmp = esti_depth;
%clip_min = 200;
%clip_max = 260;
%esti_depth_tmp(esti_depth_tmp < clip_min) = clip_min;
%esti_depth_tmp(esti_depth_tmp > clip_max) = clip_max;
imagesc(esti_depth_tmp,'AlphaData', ~nan_mask);
title('Estimated depth');
set(gcf,'color','w');
colorbar('southoutside');

imwrite(color_I1, [ambi_remove_dir, '/left_view.png']);
imwrite( ind2rgb(im2uint8(mat2gray(esti_depth_tmp)), parula(256)), [ambi_remove_dir, '/depth_esti.png'], 'png', 'Alpha', uint8(~nan_mask) * 255);
function showchesspoint(imgi,left_test)
    figure;
    imshow(imgi); 
    hold on;    
    i = 1;
    while i < 55
        plot(left_test(i,1), left_test(i,2), 'ro', 'MarkerSize', 5); 
        i = i+1;
    end
    %plot(right_test(1), right_test(2), 'bo', 'MarkerSize', 5);
    %plot([left_test(1), right_test(1)], [left_test(2), right_test(2)], 'k-', 'LineWidth', 1); % 黑色线
end
function showMatchesdist(imgi,left_test,right_test)
    figure;
    imshow(imgi); 
    hold on;    
    plot(left_test(1), left_test(2), 'ro', 'MarkerSize', 5); 
    plot(right_test(1), right_test(2), 'bo', 'MarkerSize', 5);
    plot([left_test(1), right_test(1)], [left_test(2), right_test(2)], 'k-', 'LineWidth', 1); % 黑色线
end
function showMatchesSIFT(imj,imk,matchedPoints1,matchedPoints2)
    imshow1 = cat(2, imj, imk);
    figure
    imshow(imshow1);hold on;
    plot(matchedPoints1(1,:),matchedPoints1(2,:), 'ro','MarkerSize',5);
    plot(matchedPoints2(1,:)+size(imj,2),matchedPoints2(2,:), 'bo','MarkerSize',5);
    shift = size(imj,2);
    cmap = jet(32);
    k = 1;
    x1h = matchedPoints1(1:2,:);
    x2h = matchedPoints2(1:2,:);
    for ii = 1:size(x1h,2)
        ptdraw = [x1h(2,ii), x1h(1,ii);
                  x2h(2,ii), x2h(1,ii)+shift];
        plot(ptdraw(:,2),ptdraw(:,1),'LineStyle','-','LineWidth',0.5,'Color',cmap(k,:));
        k = mod(k+1,32);if k == 0, k = 1;end
    end
end
%==========================================================================





