function FK = robot_FK(body_coords, q, l, a)

    % generate homogenous transforms
    H_o_b = homog([body_coords(1); body_coords(2)],...
                   twoDRotate(body_coords(3))); % body w.r.t origin
    H_b_h1 = homog([0; -a/2], twoDRotate(q(1))); % hinge w.r.t body
    H_b_h3 = homog([0; -a/2], twoDRotate(q(3))); % hinge w.r.t body

    %joints
    H_h1_1 = homog([0; -l], twoDRotate(q(2))); 
    H_1_2 = homog([0; -l], twoDRotate(0)); 

    H_h3_3 = homog([0; -l], twoDRotate(q(4))); 
    H_3_4 = homog([0; -l], twoDRotate(0)); 

    FK = struct();
    %leg1
    FK.H_o_h1 = H_o_b * H_b_h1; % hinge1 w.r.t world
    FK.H_o_1 = FK.H_o_h1 * H_h1_1; % p1 w.r.t world
    FK.H_o_2 = FK.H_o_1 * H_1_2; % p2 w.r.t world
    
    %leg2
    FK.H_o_h3 = H_o_b * H_b_h3; % hinge3 w.r.t world 
    FK.H_o_3 = FK.H_o_h3 * H_h3_3; % p3 w.r.t world
    FK.H_o_4 = FK.H_o_3 * H_3_4; % p4 w.r.t world

    % extract positions
    FK.hinge = extract_pos(FK.H_o_h1);
    FK.p1 = extract_pos(FK.H_o_1);
    FK.p2 = extract_pos(FK.H_o_2);
    FK.p3 = extract_pos(FK.H_o_3);
    FK.p4 = extract_pos(FK.H_o_4);

    % nested helper functions
    function rotm = twoDRotate(q)
        rotm = [cos(q), -sin(q);
                sin(q), cos(q)];
    end

    function H = homog(d, r)
    % homogenous transform for 2D kinematics, NOTE: Rotate then translate

    trans = [eye(2), d;
            zeros(1, 2), 1];
    rot = [r, zeros(2,1);
           zeros(1,2), 1];
    H = trans * rot;
    end

    function pos = extract_pos(H)
        pos = [H(1, 3); H(2, 3)];
    end
end