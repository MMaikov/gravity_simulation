#pragma once

#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>

struct video_encoder {
    AVFormatContext* format_ctx;
    AVCodecContext* codec_ctx;
    AVStream* stream;
    AVPacket* packet;
    AVFrame* frame;
    int64_t pts;
};

int video_encoder_init(struct video_encoder* enc, const char* filename);
int video_encoder_send_frame(struct video_encoder* enc, const uint8_t* grayscale_data);
void video_encoder_finish(struct video_encoder* enc);