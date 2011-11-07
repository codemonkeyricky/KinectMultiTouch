#include <XnOpenNI.h>
#include <XnCppWrapper.h>

#include "cv.h"

#include "assert.h"
#include "testApp.h"
#include "armadillo"
#include <iostream>
#include <sys/timeb.h>

#define RES_X           640
#define RES_Y           480

#if 1
#define SCREEN_RES_X    1680.0
#define SCREEN_RES_Y    1050.0
#else
#define SCREEN_RES_X    1366.0
#define SCREEN_RES_Y    768.0
#endif

#undef TV
#undef DEBUG

#if defined(TV)
#define Z_DEPTH         10
#define NUDGE           30
#define POINT           15
#define BLOCK_TIMEOUT   0.3f
#define SCROLL_DIST     0.02f
#else
#define Z_DEPTH         7
#define NUDGE           10
#define POINT           10
#define BLOCK_TIMEOUT   0.3f
#define SCROLL_DIST     0.02f
#endif

using namespace arma;

// OpenNI objects.
xn::Context         g_Context;
xn::DepthGenerator  g_DepthGenerator;
xn::ImageGenerator  g_ImageGenerator;

// Depth map - captured every time update() is called.
static unsigned short   g_depth[RES_X*RES_Y];

static unsigned char    g_color[3*RES_X*RES_Y];

static int              g_x_unit_len;
static int              g_y_unit_len;

static int              g_capture_width;
static int              g_capture_height;

// Transformed map - a map of 1s and 0s.
//
// This allows us to feed into the openCV library.
static unsigned char    g_transformed[RES_X*RES_Y];

// Constant transformed offset. Added to the transformed
// map to complete the transformation.
static mat              g_offset;

// Transformation matrix - applied to all the points.
static mat              g_trans;

// A list of filtered touch points.
static vector<cv::Point2i> g_touch_points;

// Total number of points used for calibration.
//
// Need 3 to move onto the active state.
static mat              g_calibrate_points[3];
static unsigned int     g_calibrate_state_pt_count;

typedef enum
{
    CALIBRATE_RENDER_MODE_DEPTH,
    CALIBRATE_RENDER_MODE_COLOR
} eCALIBRATE_RENDER_MODE;

static eCALIBRATE_RENDER_MODE   g_calibrate_render_mode;

static unsigned char g_active_state_render;

static mat              g_capture_points[2];
static unsigned int     g_capture_points_count;

static float            g_y_scale_factor;

static float            g_scale_conv;
static float            g_z_depth_factor;

static unsigned char    mousedown;

static float            total_elapsed;
static unsigned char    block;
static unsigned char    scroll_mode;

#define SAMPLE_XML_PATH "../Sample-Tracking.xml"

// eSTATE
//  System state information
typedef enum
{
    STATE_CALIBRATE,
    STATE_ACTIVE
} eSTATE;

static eSTATE g_state;

static void calculate_transform(
    void
    );

static void determine_y_scale_factor(
    void
    );

//--------------------------------------------------------------
void testApp::setup()
{
    XnStatus                rv;
	xn::EnumerationErrors   errors;

	// Initialize OpenNI
	rv = g_Context.InitFromXmlFile(SAMPLE_XML_PATH,
                                   &errors);
    assert(rv == 0);

    // Find depth node
	rv = g_Context.FindExistingNode(XN_NODE_TYPE_DEPTH,
                                    g_DepthGenerator);
    assert(rv == 0);

    // Find image generator
    rv = g_Context.FindExistingNode(XN_NODE_TYPE_IMAGE,
                                    g_ImageGenerator);
    assert(rv == 0);

    g_DepthGenerator.GetAlternativeViewPointCap().SetViewPoint(g_ImageGenerator);

    // Kick off depth map generation
    rv = g_Context.StartGeneratingAll();
    assert(rv == 0);

    // Initializes the calibrate points.
    g_calibrate_points[0].set_size(3,1);
	g_calibrate_points[1].set_size(3,1);
	g_calibrate_points[2].set_size(3,1);

    // Initialize capture points.
    g_capture_points[0].set_size(2,1);
    g_capture_points[1].set_size(2,1);
}

void inline calibrate_state_update(
    void
    )
{
    // --------------------
    // Copy depth data
    // --------------------

    // Update depth information
    g_Context.WaitOneUpdateAll(g_DepthGenerator);

    xn::DepthMetaData depthMD;
    g_DepthGenerator.GetMetaData(depthMD);
    const XnDepthPixel* pDepth = depthMD.Data();

    // Copy depth data into local buffer
    memcpy(&g_depth[0],
           pDepth,
           RES_X*RES_Y*2);

    // --------------------
    // Copy color data
    // --------------------

    // Update color information
    g_Context.WaitOneUpdateAll(g_ImageGenerator);

    xn::ImageMetaData imageMD;
    g_ImageGenerator.GetMetaData(imageMD);
    const unsigned char *pColor = imageMD.Data();

    // Copy color data into local buffer
    memcpy(&g_color[0],
           pColor,
           3*RES_X*RES_Y);
}

// active_state_update()
//  Active state update.
void inline active_state_update(
    void
    )
{
    int width;
    int height;
    int touch_pts;

    struct timeb t_ref;
    struct timeb t_cur;
    struct timeb t_start;
    struct timeb t_end;

    // Update depth information
    g_Context.WaitOneUpdateAll(g_DepthGenerator);

    xn::DepthMetaData depthMD;
    g_DepthGenerator.GetMetaData(depthMD);
    const XnDepthPixel  *pDepth = depthMD.Data();

    ftime(&t_start);

    ftime(&t_ref);

    // Construct a matrix able to hold all points, and only copy valid points.
    int num_valid_pts = 0;

    mat points(3,
               g_capture_width*g_capture_height);

    for(int row = g_capture_points[0](1, 0);
        row < g_capture_points[1](1, 0);
        row++)
    {
        for(int column = g_capture_points[0](0, 0);
            column < g_capture_points[1](0, 0);
            column++)
        {
            if(pDepth[row*RES_X + column] != 0)
            {
                points(0, num_valid_pts) = column;
                points(1, num_valid_pts) = row;
                points(2, num_valid_pts) = pDepth[row*RES_X + column];

                num_valid_pts++;
            }
        }
    }

    ftime(&t_cur);

    printf("t2: %u\n",
           t_cur.millitm - t_ref.millitm);

    t_ref = t_cur;

    // Apply transformation to the valid points
    mat pointsT = g_trans * points;

    ftime(&t_cur);

    printf("t3: %u\n",
           t_cur.millitm - t_ref.millitm);

    t_ref = t_cur;

    // Apply offset to the translated points
    pointsT = pointsT + g_offset;

    ftime(&t_cur);

    printf("t5: %u\n",
           t_cur.millitm - t_ref.millitm);

    t_ref = t_cur;

    // Determine how many points fall within (1, 1, 0.1) after
    // transformation

    memset(&g_transformed[0],
           0,
           RES_X*RES_Y);

    for(int i = 0; i < num_valid_pts; i++)
    {
        if(!(   (pointsT(0, i) > 0)
             && (pointsT(1, i) > 0)
             && (pointsT(2, i) > 0)))
        {
            continue;
        }

        if(!(   (pointsT(0, i) < 1)
             && (pointsT(1, i) < g_y_scale_factor)
             && (pointsT(2, i) < g_z_depth_factor)))
        {
            continue;
        }

        // Clear transformation
        int x = pointsT(0, i) * g_x_unit_len;
        int y = (g_y_scale_factor - pointsT(1, i)) * g_y_unit_len;

        g_transformed[y*RES_X + x] = 1;
    }

    ftime(&t_cur);

    printf("t6: %u\n",
           t_cur.millitm - t_ref.millitm);

    t_ref = t_cur;

    // Construct a binary image
    cv::Mat img(RES_Y,
                RES_X,
                CV_8UC1);
    unsigned char *data = img.data;

    memcpy(data,
           &g_transformed[0],
           RES_X*RES_Y);

    vector< vector<cv::Point2i> > contours;
    findContours(img,
                 contours,
                 CV_RETR_LIST,
                 CV_CHAIN_APPROX_SIMPLE);

    touch_pts = 0;
    g_touch_points.clear();
    for(int i = 0; i < contours.size(); i++)
    {
        cv::Mat contourMat(contours[i]);
        if(contourArea(contourMat) > POINT)
        {
            cv::Scalar center = mean(contourMat);
            printf("Point %d: x = %d, y = %d\n",
                    i,
                    (int)center[0],
                    (int)center[1]);

            cv::Point2i new_pt;
            new_pt.x = center[0];
            new_pt.y = center[1];

            g_touch_points.push_back(new_pt);

            touch_pts++;
        }
    }

    ftime(&t_cur);

    printf("t7: %u\n",
           t_cur.millitm - t_ref.millitm);

    t_ref = t_cur;

    float elapsed = ofGetElapsedTimef();
    if((elapsed - total_elapsed) > BLOCK_TIMEOUT)
    {
        block = 0;
        total_elapsed = elapsed;
    }

    char command[50];
    if(g_touch_points.size() == 1)
    {
        int i = 0;

        float x_pos = g_touch_points[i].x * 1.0f / g_x_unit_len;
        float y_pos = g_touch_points[i].y * 1.0f / g_y_scale_factor / g_y_unit_len;

        printf("click pos: (%f, %f)\n",
               x_pos,
               y_pos);

        unsigned int x_click = x_pos * SCREEN_RES_X;
        unsigned int y_click = y_pos * SCREEN_RES_Y;

        printf("click: (%u, %u)\n",
               x_click,
               y_click);

        if(block == 0)
        {
            sprintf(command,
                    "xdotool mousemove %d %d click 1",
                    x_click,
                    y_click);

#if !defined(DEBUG)
            system(command);
#endif

            block = 1;
        }
    }

    static cv::Point2f scroll_ref;
    if(g_touch_points.size() >= 2)
    {
        static cv::Point2f temp;
        static cv::Point2f temp2;

        if(scroll_mode == 0)
        {
            scroll_ref.x = g_touch_points[0].x * 1.0f / g_x_unit_len;
            scroll_ref.y = g_touch_points[0].y * 1.0f / g_y_scale_factor / g_y_unit_len;

            scroll_mode = 1;
        }
        else
        {
            temp.x = g_touch_points[0].x * 1.0f / g_x_unit_len;
            temp.y = g_touch_points[0].y * 1.0f / g_y_scale_factor / g_y_unit_len;

            temp2.x = temp.x - scroll_ref.x;
            temp2.y = temp.y - scroll_ref.y;

            if(temp2.y > SCROLL_DIST)
            {
#if !defined(DEBUG)
                system("xdotool click 5");
#endif
                // scroll_ref  = temp;
            }
            else if(temp2.y < (-SCROLL_DIST))
            {
#if !defined(DEBUG)
                system("xdotool click 4");
#endif
                // scroll_ref  = temp;
            }
        }
    }

    if(g_touch_points.size() < 2)
    {
        scroll_mode = 0;
    }

#if 0
    for(int i = 0; i < g_touch_points.size(); i++)
    {
        float x_pos = g_touch_points[i].x * 1.0f / g_x_unit_len;
        float y_pos = g_touch_points[i].y * 1.0f / g_y_scale_factor / g_y_unit_len;

        printf("click pos: (%f, %f)\n",
               x_pos,
               y_pos);

        unsigned int x_click = x_pos * SCREEN_RES_X;
        unsigned int y_click = y_pos * SCREEN_RES_Y;

        printf("click: (%u, %u)\n",
               x_click,
               y_click);

        if(block == 0)
        {
            sprintf(command,
                    "xdotool mousemove %d %d click 1",
                    x_click,
                    y_click);

            system(command);

            block = 1;
        }

#if 0
        if(mousedown == 0)
        {
            sprintf(command,
                    "xdotool mousedown 1");

            system(command);

            mousedown = 1;
        }
#endif
    }

#if 0
    if(g_touch_points.size() == 0)
    {
        sprintf(command,
                "xdotool mouseup 1");

        system(command);

        mousedown = 0;
    }
#endif
#endif

    ftime(&t_end);

    printf("t_t: %u\n",
           t_end.millitm - t_start.millitm);

    printf("Touch Points: %d\n",
            touch_pts);
}

//--------------------------------------------------------------
void testApp::update()
{
    // Handle it accordingly.
    switch(g_state)
    {
        case STATE_CALIBRATE:

            calibrate_state_update();
            break;

        case STATE_ACTIVE:

            active_state_update();
            break;

        default:
            assert(0);
    }
}

// calibrate_state_draw()
//  Draw function in the calibrate state.
void inline calibrate_state_draw(
    void
    )
{
    unsigned int level = 0;
    unsigned int color = 0;
    unsigned char *color_ptr;
    unsigned short *depth_ptr;


    ofFill();
    if(g_calibrate_render_mode == CALIBRATE_RENDER_MODE_DEPTH)
    {
        for(int i = 0; i < RES_X; i++)
        {
            for(int j = 0; j < RES_Y; j++)
            {
                depth_ptr = &g_depth[j*RES_X + i];
                color = (*depth_ptr << 16) | (*depth_ptr << 8) | (*depth_ptr);

                ofSetColor(color);

                ofRect(i, j, 1, 1);
            }
        }
    }
    else // CALIBRATE_RENDER_MODE_COLOR
    {
        for(int i = 0; i < RES_X; i++)
        {
            for(int j = 0; j < RES_Y; j++)
            {
                color_ptr = &g_color[j*RES_X*3 + i*3];
                color = (*color_ptr << 16) | (*(color_ptr+1) << 8) | *(color_ptr + 2);

                ofSetColor(color);

                ofRect(i, j, 1, 1);
            }
        }
    }

    ofNoFill();
    ofSetColor(0xff0000);
    if(g_calibrate_state_pt_count >= 1)
    {
        // Draw the first point
        ofCircle(g_calibrate_points[0](0, 0),
                 g_calibrate_points[0](1, 0),
                 10);
    }

    if(g_calibrate_state_pt_count >= 2)
    {
        // Draw the second point
        ofCircle(g_calibrate_points[1](0, 0),
                 g_calibrate_points[1](1, 0),
                 10);

        // Draw the first line
        ofLine(g_calibrate_points[0](0, 0),
               g_calibrate_points[0](1, 0),
               g_calibrate_points[1](0, 0),
               g_calibrate_points[1](1, 0));
    }

    if(g_calibrate_state_pt_count >= 3)
    {
        // Draw the third point
        ofCircle(g_calibrate_points[2](0, 0),
                 g_calibrate_points[2](1, 0),
                 10);

        // Draw the second line
        ofLine(g_calibrate_points[0](0, 0),
               g_calibrate_points[0](1, 0),
               g_calibrate_points[2](0, 0),
               g_calibrate_points[2](1, 0));
    }

    if(g_capture_points_count >= 1)
    {
        // Draw the third point
        ofCircle(g_capture_points[0](0, 0),
                 g_capture_points[0](1, 0),
                 5);
    }

    if(g_capture_points_count >= 2)
    {
        // Draw the third point
        ofCircle(g_capture_points[1](0, 0),
                 g_capture_points[1](1, 0),
                 5);

        // Draw the second line
        ofLine(g_capture_points[0](0, 0),
               g_capture_points[0](1, 0),
               g_capture_points[1](0, 0),
               g_capture_points[1](1, 0));
    }
}

// active_state_draw()
//  Draw function in the active state.
void inline active_state_draw(
    void
    )
{
    unsigned int color;
    unsigned int level;


    ofSetColor(0x000000);

    if(g_active_state_render)
    {
        for(int i = 0; i < RES_X; i++)
        {
            for(int j = 0; j < RES_Y; j++)
            {
                ofFill();
                level = g_transformed[j*RES_X + i];

                if(level != 0)
                {
                    level = 0xFF;
                }

                color = level | (level << 8) | (level << 16);
                ofSetColor(color);

                ofRect(i, j, 1, 1);
            }
        }
    }
}

//--------------------------------------------------------------
void testApp::draw()
{
    switch(g_state)
    {
        case STATE_CALIBRATE:

            calibrate_state_draw();
            break;

        case STATE_ACTIVE:

            active_state_draw();
            break;

        default:
            assert(0);
    }
}

//--------------------------------------------------------------
void testApp::keyPressed(int key){

    if(key == 'q')
    {
        // Right mouse click

        // calculate transformation and move to active state.

        if(g_calibrate_state_pt_count < 3)
        {
            // Need 3 points to move to the active state.
            return;
        }

        if(g_capture_points_count < 2)
        {
            // Need 3 points to move to the active state.
            return;
        }

        // Calculate transform matrix and advance to active state.
        calculate_transform();

        determine_y_scale_factor();

        // Move to active state.
        g_state = STATE_ACTIVE;
    }

    if(key == 'w')
    {
        if(g_calibrate_render_mode == CALIBRATE_RENDER_MODE_DEPTH)
        {
            g_calibrate_render_mode = CALIBRATE_RENDER_MODE_COLOR;
        }
        else
        {
            g_calibrate_render_mode = CALIBRATE_RENDER_MODE_DEPTH;
        }
    }

    if(key == 'e')
    {
        g_active_state_render = !g_active_state_render;
    }
}

//--------------------------------------------------------------
void testApp::keyReleased(int key){

}

//--------------------------------------------------------------
void testApp::mouseMoved(int x, int y ){

}

//--------------------------------------------------------------
void testApp::mouseDragged(int x, int y, int button){

}

// calculate_transform()
//  Calculate transform based on selected calibrated points.
//
void calculate_transform(
    void
    )
{
    int nudge = NUDGE;
    mat transform;
    mat inv_transform;
    int width;
    int height;

    g_calibrate_points[0](2, 0) -= nudge;
    g_calibrate_points[1](2, 0) -= nudge;
    g_calibrate_points[2](2, 0) -= nudge;

    // offsets at bottom left
    mat translation = g_calibrate_points[0];

    // determine the new x-axis
    mat x_axis = g_calibrate_points[1] - g_calibrate_points[0];
    int scale = norm(x_axis, 1);
    g_scale_conv = scale;

    // determine a planar vector for the x-axis
    mat planar_vec = g_calibrate_points[2] - g_calibrate_points[0];

    // determine the z-axis by doing a cross product between
    // x-axis and the planar vector
    mat z_axis = cross(x_axis, planar_vec);
    z_axis = z_axis / norm(z_axis, 1) * scale;

    // determine the y-axis by doing a cross product between
    // the x-axis and the z-axis
    mat y_axis = cross(x_axis, z_axis);
    y_axis = y_axis / norm(y_axis, 1) * scale * -1;

    mat rot_scale(3,3);
    rot_scale(0, 0) = x_axis(0, 0);
    rot_scale(0, 1) = x_axis(1, 0);
    rot_scale(0, 2) = x_axis(2, 0);

    rot_scale(1, 0) = y_axis(0, 0);
    rot_scale(1, 1) = y_axis(1, 0);
    rot_scale(1, 2) = y_axis(2, 0);

    rot_scale(2, 0) = z_axis(0, 0);
    rot_scale(2, 1) = z_axis(1, 0);
    rot_scale(2, 2) = z_axis(2, 0);

    // Find the transpose
    rot_scale = trans(rot_scale);

    // Construct transform matrix

    // Start with identity matrix
    transform = eye<mat>(4,4);

    // Set the rotation portion of the matrix
    for(int i = 0; i < 3; i++)
    {
        for(int j = 0; j < 3; j++)
        {
            transform(i, j) = rot_scale(i, j);
            transform(i, j) = rot_scale(i, j);
            transform(i, j) = rot_scale(i, j);
        }
    }

    // Set the translation portion of the matrix
    transform(0, 3) = translation(0, 0);
    transform(1, 3) = translation(1, 0);
    transform(2, 3) = translation(2, 0);

    // Get inverse transform
    inv_transform = inv(transform);

    g_capture_width  = g_capture_points[1](0, 0) - g_capture_points[0](0, 0);
    g_capture_height = g_capture_points[1](1, 0) - g_capture_points[0](1, 0);

    // Construct the offset matrix.
    g_offset = mat(3, g_capture_width*g_capture_height);
    g_offset.row(0).fill(inv_transform(0, 3));
    g_offset.row(1).fill(inv_transform(1, 3));
    g_offset.row(2).fill(inv_transform(2, 3));

    // Construct transform matrix.
    g_trans = inv_transform.submat(0, 0, 2, 2);

    width = g_calibrate_points[1](0, 0) - g_calibrate_points[0](0, 0);
    height = g_calibrate_points[1](1, 0) - g_calibrate_points[0](1, 0);
    g_x_unit_len = sqrt(width*width + height*height);

    width = g_calibrate_points[2](0, 0) - g_calibrate_points[0](0, 0);
    height = g_calibrate_points[2](1, 0) - g_calibrate_points[0](1, 0);
    g_y_unit_len = sqrt(width*width + height*height);

    g_z_depth_factor = Z_DEPTH * 1.0f / scale;
}

void determine_y_scale_factor(
    void
    )
{
    // Construct a matrix able to hold all points, and only copy valid points.
    int num_valid_pts = 1;
    mat points(3, g_capture_width*g_capture_height);

    points(0, 0) = g_calibrate_points[2](0, 0);
    points(1, 0) = g_calibrate_points[2](1, 0);
    points(2, 0) = g_calibrate_points[2](2, 0);

    // DEBUG
    points(0, 1) = g_calibrate_points[1](0, 0);
    points(1, 1) = g_calibrate_points[1](1, 0);
    points(2, 1) = g_calibrate_points[1](2, 0);

    // Resize the matrix to the correct size

    mat pointsT = g_trans * points;

    // Apply offset to the translated points
    // pointsT = pointsT + offset;
    pointsT = pointsT + g_offset;

    // Get y scale factor
    g_y_scale_factor = pointsT(1, 0);   // point 0, y

#if 0
    float debug_1 = pointsT(0, 1);        // point 1, x
    float debug_2 = pointsT(1, 1);        // point 1, y

    debug_1 = debug_1 + 1 - 1;
#endif
}

void calibrate_state_mousePressed(
    int x,
    int y,
    int button
    )
{
    int depth;
    int index;


    switch(button)
    {
        case 0:
            // left mouse click

            // record selected point.

            if(g_calibrate_state_pt_count >= 3)
            {
                // Already have 3 points, return.
                return;
            }

            depth = g_depth[y*RES_X + x];
            if(depth == 0)
            {
                // Invalid point. Need to re-select.
                return;
            }

            // Record current point.
            g_calibrate_points[g_calibrate_state_pt_count](0, 0) = x;
            g_calibrate_points[g_calibrate_state_pt_count](1, 0) = y;
            g_calibrate_points[g_calibrate_state_pt_count](2, 0) = depth;

            // Advance state.
            g_calibrate_state_pt_count++;
            break;

        case 1:
            // Middle mouse click

            // Restart calibration.
            g_calibrate_state_pt_count  = 0;
            g_capture_points_count      = 0;
            break;

        case 2:

            if(g_capture_points_count >= 2)
            {
                // Already have 2 points, return.
                return;
            }

            g_capture_points[g_capture_points_count](0, 0) = x;
            g_capture_points[g_capture_points_count](1, 0) = y;

            g_capture_points_count++;
            break;

        default:
            assert(0);
    }
}

void active_state_mousePressed(
    int button
    )
{
    switch(button)
    {
        case 0:

            // left mouse click.

            // Bump back to calibrate state.
            g_state = STATE_CALIBRATE;
            break;

        case 1:
            break;

        case 2:
            break;

        default:
            break;
            // assert(0);
    }
}

//--------------------------------------------------------------
void testApp::mousePressed(int x, int y, int button)
{
    switch(g_state)
    {
        case STATE_CALIBRATE:

            calibrate_state_mousePressed(x, y, button);
            break;

        case STATE_ACTIVE:

            active_state_mousePressed(button);
            break;

        default:
            assert(0);
    }
}

//--------------------------------------------------------------
void testApp::mouseReleased(int x, int y, int button){

}

//--------------------------------------------------------------
void testApp::windowResized(int w, int h){

}

