clear
clc

scene_list = dir('landmark/*.csv');

for i = 1:length(scene_list)
    scene_id = str2double(scene_list(i).name(1:strfind(scene_list(i).name,'_')-1));

    frame_num = 0;
    landmarks = readtable(strcat(num2str(scene_id),'/',num2str(scene_id),'_landmarks.csv'));
    landmarks = table2array(landmarks);
    frame_list = landmarks(:,2);
    tracks = readtable(strcat(num2str(scene_id),'/',num2str(scene_id),'_tracks.csv'));
    tracks = table2array(tracks);
    tracksMeta = readtable(strcat(num2str(scene_id),'/',num2str(scene_id),'_trackMeta.csv'));
    map_seg = readtable(strcat(num2str(scene_id),'/',num2str(scene_id),'_mapSegmentation.csv'));

    ped_ids = table2array(tracksMeta(ismember(tracksMeta.class, 'pedestrian'),2));
    parked_car_ids = table2array(tracksMeta(ismember(tracksMeta.class, 'parked car'),2));
    car_ids = table2array(tracksMeta(ismember(tracksMeta.class, 'car'),2));
    bicycle_ids = table2array(tracksMeta(ismember(tracksMeta.class, 'bicycle'),2));

    recordingMeta = readtable(strcat(num2str(scene_id),'/',num2str(scene_id),'_recordingMeta.csv'));
    px2meter = recordingMeta.px2meter;

    v = VideoWriter(strcat(num2str(scene_id),'/',num2str(scene_id),'_result'));
    open(v)
    fig = figure();
    for i = 1:length(frame_list)
        frame_id = frame_list(i);
        frame_id_for_load = frame_id + 1;
        image = imread(strcat('./',num2str(scene_id),'/images/',num2str(scene_id),'_',num2str(frame_id_for_load,'%04d'),'.jpg'));
        imshow(image)
        hold on

        row_in_frame = tracks(tracks(:,3)==frame_id,:);
        [~,ped_row_in_frame] = intersect(row_in_frame(:,2),ped_ids);
        [~,parked_car_row_in_frame] = intersect(tracks(:,2),parked_car_ids);
        [~,car_in_frame] = intersect(row_in_frame(:,2),car_ids);
        [~,bicycle_in_frame] = intersect(row_in_frame(:,2),bicycle_ids);

        scatter(row_in_frame(ped_row_in_frame,5)/px2meter ,-row_in_frame(ped_row_in_frame,6)/px2meter,'r')
        scatter(row_in_frame(bicycle_in_frame,5)/px2meter ,-row_in_frame(bicycle_in_frame,6)/px2meter,'b')
        for j = 1:length(car_in_frame)
            heading_deg = row_in_frame(car_in_frame(j),7);
            x_pos_meter = row_in_frame(car_in_frame(j),5);
    %         추후에 y_pos_meter를 계산할때에 -값을 빼야됨
            y_pos_meter = row_in_frame(car_in_frame(j),6);
            car_length_meter = row_in_frame(car_in_frame(j),9);
            car_width_meter = row_in_frame(car_in_frame(j),8);
            pgon_meter = polyshape([x_pos_meter-0.5*car_length_meter*cosd(heading_deg)-0.5*car_width_meter*sind(heading_deg),...
                                    x_pos_meter-0.5*car_length_meter*cosd(heading_deg)+0.5*car_width_meter*sind(heading_deg),...
                                    x_pos_meter+0.5*car_length_meter*cosd(heading_deg)+0.5*car_width_meter*sind(heading_deg),...
                                    x_pos_meter+0.5*car_length_meter*cosd(heading_deg)-0.5*car_width_meter*sind(heading_deg)],...
                                    [y_pos_meter-0.5*car_length_meter*sind(heading_deg)+0.5*car_width_meter*cosd(heading_deg),...
                                    y_pos_meter-0.5*car_length_meter*sind(heading_deg)-0.5*car_width_meter*cosd(heading_deg),...
                                    y_pos_meter+0.5*car_length_meter*sind(heading_deg)-0.5*car_width_meter*cosd(heading_deg),...
                                    y_pos_meter+0.5*car_length_meter*sind(heading_deg)+0.5*car_width_meter*cosd(heading_deg)]);
            pgon_meter.Vertices = [pgon_meter.Vertices(:,1)/px2meter, -pgon_meter.Vertices(:,2)/px2meter];
            plot(pgon_meter,'FaceColor','c','FaceAlpha',0.5)
        end

        for j = 1:length(parked_car_row_in_frame)
            heading_deg = tracks(parked_car_row_in_frame(j),7);
            x_pos_meter = tracks(parked_car_row_in_frame(j),5);
    %         추후에 y_pos_meter를 계산할때에 -값을 빼야됨
            y_pos_meter = tracks(parked_car_row_in_frame(j),6);
            car_length_meter = tracks(parked_car_row_in_frame(j),9);
            car_width_meter = tracks(parked_car_row_in_frame(j),8);
            pgon_meter = polyshape([x_pos_meter-0.5*car_length_meter*cosd(heading_deg)-0.5*car_width_meter*sind(heading_deg),...
                                    x_pos_meter-0.5*car_length_meter*cosd(heading_deg)+0.5*car_width_meter*sind(heading_deg),...
                                    x_pos_meter+0.5*car_length_meter*cosd(heading_deg)+0.5*car_width_meter*sind(heading_deg),...
                                    x_pos_meter+0.5*car_length_meter*cosd(heading_deg)-0.5*car_width_meter*sind(heading_deg)],...
                                    [y_pos_meter-0.5*car_length_meter*sind(heading_deg)+0.5*car_width_meter*cosd(heading_deg),...
                                    y_pos_meter-0.5*car_length_meter*sind(heading_deg)-0.5*car_width_meter*cosd(heading_deg),...
                                    y_pos_meter+0.5*car_length_meter*sind(heading_deg)-0.5*car_width_meter*cosd(heading_deg),...
                                    y_pos_meter+0.5*car_length_meter*sind(heading_deg)+0.5*car_width_meter*cosd(heading_deg)]);
            pgon_meter.Vertices = [pgon_meter.Vertices(:,1)/px2meter, -pgon_meter.Vertices(:,2)/px2meter];
            plot(pgon_meter,'FaceColor','y','FaceAlpha',0.5)
        end
    %     saveas(fig,strcat('./샘플_result/images/',num2str(frame_id),'.jpg'));
        writeVideo(v,getframe(gcf))
        pause(0.1)

        hold off
    end
    close(v)
end
