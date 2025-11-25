#!/usr/bin/env python


# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv

from numpy import pi, sin, cos


# Default size for a generated background image when none is provided
defaultSize = 512

class TestSceneRender():
    """A simple synthetic scene renderer for testing.

    This class can render a background (either provided or a blank image)
    and optionally a foreground image that oscillates over time. It is
    intended for unit tests or demos where a moving object/rect is needed.
    """

    def __init__(self, bgImg=None, fgImg=None, deformation=False, speed=0.25, **params):
        # Current simulated time (seconds)
        self.time = 0.0
        # Time increment per frame (seconds) -- default 30 FPS
        self.timeStep = 1.0 / 30.0
        # Foreground image (if any) placed on top of background
        self.foreground = fgImg
        # Whether to apply deformation when rendering a rectangle instead
        # of a foreground image
        self.deformation = deformation
        # Oscillation speed multiplier for motion
        self.speed = speed

        # If a background image was provided, copy it. Otherwise create a
        # blank color image of size defaultSize x defaultSize.
        if bgImg is not None:
            self.sceneBg = bgImg.copy()
        else:
            # Create a blank 3-channel image to use as the background
            self.sceneBg = np.zeros((defaultSize, defaultSize, 3), np.uint8)

        # store height and width for convenience (note: shape indices)
        self.h = self.sceneBg.shape[0]  # Height (rows)
        self.w = self.sceneBg.shape[1]  # Width (cols)

        # If a foreground image is provided, compute an initial center so the
        # foreground is roughly centered in the background. We also compute
        # amplitude limits for x and y oscillation so the foreground does not
        # move outside the background image bounds.
        if fgImg is not None:
            self.foreground = fgImg.copy()
            # center expressed as (row, col) to match numpy indexing
            self.center = self.currentCenter = (int(self.h / 2 - fgImg.shape[0] / 2), int(self.w / 2 - fgImg.shape[1] / 2))

            # How far the foreground is allowed to move from the initial center
            self.xAmpl = self.w - (self.center[1] + fgImg.shape[1])
            self.yAmpl = self.h - (self.center[0] + fgImg.shape[0])

        # initial rectangle corners (used when no foreground image is provided)
        # The rectangle is defined by its four corners: top-left, top-right,
        # bottom-right, bottom-left. Values are stored in (row, col) order
        # consistent with numpy / OpenCV coordinate usage.
        self.initialRect = np.array([
            (self.h / 2, self.w / 2),
            (self.h / 2, self.w / 2 + self.w / 10),
            (self.h / 2 + self.h / 10, self.w / 2 + self.w / 10),
            (self.h / 2 + self.h / 10, self.w / 2)
        ]).astype(int)

        # currentRect starts as the initial rect and can be modified over time
        self.currentRect = self.initialRect

    def getXOffset(self, time):
        """Return the horizontal offset (in pixels) at a given time.

        The offset is computed as a cosine oscillation scaled by x amplitude
        and the configured speed.
        """
        return int(self.xAmpl * cos(time * self.speed))

    def getYOffset(self, time):
        """Return the vertical offset (in pixels) at a given time.

        The offset is computed as a sine oscillation scaled by y amplitude
        and the configured speed.
        """
        return int(self.yAmpl * sin(time * self.speed))

    def setInitialRect(self, rect):
        """Replace the initial rectangle used when rendering without a
        foreground image.
        """
        self.initialRect = rect

    def getRectInTime(self, time):
        """Return the bounding rectangle [y0, x0, y1, x1] of the moving
        foreground (or initial rect) at the provided time.

        For a foreground image the rectangle is computed from the center
        plus the current offsets. When no foreground image is present the
        initialRect is offset instead.
        """
        if self.foreground is not None:
            # temporary variable to calculate the new center with offsets
            tmp = np.array(self.center) + np.array((self.getXOffset(time), self.getYOffset(time)))
            x0, y0 = tmp
            x1, y1 = tmp + np.array((self.foreground.shape[1], self.foreground.shape[0]))  # (col, row)
            return np.array([y0, x0, y1, x1])
        else:
            # Update and return bounding box for the rectangle
            x0, y0 = self.initialRect[0] + np.array((self.getXOffset(time), self.getYOffset(time)))
            x1, y1 = self.initialRect[2] + np.array((self.getXOffset(time), self.getYOffset(time)))
            return np.array([y0, x0, y1, x1])

    def getCurrentRect(self):
        """Return the current rectangle (y0, x0, y1, x1) based on the
        current state. This does not advance time.
        """
        if self.foreground is not None:
            # Calculate the bounding box of the foreground image (row/col coords)
            row0, col0 = self.currentCenter
            row1 = row0 + self.foreground.shape[0]
            col1 = col0 + self.foreground.shape[1]
            return np.array([row0, col0, row1, col1])
        else:
            # Return bounding box for the polygon (stored as (row,col) pairs)
            row0, col0 = self.currentRect[0]
            row1, col1 = self.currentRect[2]
            return np.array([row0, col0, row1, col1])

    def getNextFrame(self):
        """Advance the simulated time by one timestep and render the next
        frame.

        If a foreground image is set, it will be copied into the background
        at the current center position. If no foreground is present, a filled
        convex polygon (rectangle) is drawn instead.
        """
        img = self.sceneBg.copy()

        if self.foreground is not None:
            # Update the center position according to oscillation functions
            self.currentCenter = (self.center[0] + self.getYOffset(self.time), 
                                  self.center[1] + self.getXOffset(self.time))
            # Place the foreground image directly onto the background
            img[self.currentCenter[0]:self.currentCenter[0] + self.foreground.shape[0],
                self.currentCenter[1]:self.currentCenter[1] + self.foreground.shape[1]] = self.foreground
        else:
            # For the polygon-based object, animate the vertex coordinates
            self.currentRect = self.initialRect + np.int(30 * cos(self.time * self.speed) + 50 * sin(self.time * self.speed))
            if self.deformation:
                # Apply small oscillatory deformation to the rectangle
                self.currentRect[1:3] += int(self.h / 20 * cos(self.time))
            # Draw a filled convex polygon
            cv.fillConvexPoly(img, self.currentRect, (0, 0, 255))

        # Advance the simulation time
        self.time += self.timeStep
        return img

    def resetTime(self):
        """Reset the simulated time back to zero."""
        self.time = 0.0


def main():
    # Load example images using OpenCV's samples helper.
    backGr = cv.imread(cv.samples.findFile('graf1.png'))
    fgr = cv.imread(cv.samples.findFile('box.png'))

    # Initialize the TestSceneRender with background and foreground images
    render = TestSceneRender(backGr, fgr)

    # Simple display loop showing the animated frames until ESC is pressed
    while True:
        img = render.getNextFrame()
        cv.imshow('img', img)

        ch = cv.waitKey(3)
        if ch == 27:  # Exit on ESC key
            break

    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
