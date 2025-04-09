#include "video_encoder.h"

#include "config.h"

#include <SDL.h>
#include <libavutil/opt.h>

int video_encoder_init(struct video_encoder* enc, const char* filename) {

    av_log_set_level(AV_LOG_ERROR);

    int r = avformat_alloc_output_context2(&enc->format_ctx, NULL, "matroska", filename);
    if (r < 0) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "Could not allocate format context\n%s\n", av_err2str(r));
        return -1;
    }

    const AVCodec *codec = avcodec_find_encoder(AV_CODEC_ID_H264);
    if (!codec) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "H.264 codec not found\n");
        return -1;
    }

    enc->stream = avformat_new_stream(enc->format_ctx, NULL);
    if (!enc->stream) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "Could not create new codec stream\n");
        return -1;
    }

    enc->codec_ctx = avcodec_alloc_context3(codec);
    if (!enc->codec_ctx) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "Could not allocate codec context\n");
        return -1;
    }

    enc->codec_ctx->codec_id = AV_CODEC_ID_H264;
    enc->codec_ctx->width = WINDOW_WIDTH;
    enc->codec_ctx->height = WINDOW_HEIGHT;
    enc->codec_ctx->pix_fmt = AV_PIX_FMT_YUV420P;
    enc->codec_ctx->time_base = (AVRational){1, RECORDING_FPS};
    enc->codec_ctx->framerate = (AVRational){RECORDING_FPS, 1};
    enc->codec_ctx->gop_size = 12;
    enc->codec_ctx->max_b_frames = 1;
    enc->codec_ctx->extradata_size = 32;
    enc->codec_ctx->extradata = av_mallocz(32);
    if (!enc->codec_ctx->extradata) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "Could not allocate codec context's extradata buffer\n");
        return -1;
    }
    r = av_opt_set(enc->codec_ctx->priv_data, "preset", "veryfast", 0);
    if (r != 0) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "Could not set preset\n%s\n", av_err2str(r));
    }

    r = avcodec_open2(enc->codec_ctx, codec, NULL);
    if (r < 0) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "Could not open codec\n%s\n", av_err2str(r));
        return -1;
    }

    r = avcodec_parameters_from_context(enc->stream->codecpar, enc->codec_ctx);
    if (r < 0) {
        SDL_LogError(SDL_LOG_CATEGORY_ERROR, "Could not copy codec parameters\n%s\n", av_err2str(r));
        return -1;
    }
    enc->stream->time_base = enc->codec_ctx->time_base;

    if (!(enc->format_ctx->oformat->flags & AVFMT_NOFILE)) {
        if ((r = avio_open(&enc->format_ctx->pb, filename, AVIO_FLAG_WRITE)) < 0) {
            SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "Could not open output file\n%s\n", av_err2str(r));
            return -1;
        }
    }

    if ((r = avformat_write_header(enc->format_ctx, NULL)) < 0) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "Error writing header\n%s\n", av_err2str(r));
        return -1;
    }

    enc->frame = av_frame_alloc();
    if (!enc->frame) {
        SDL_LogError(SDL_LOG_CATEGORY_ERROR, "Could not allocate frame buffer\n");
        return -1;
    }
    enc->frame->format = enc->codec_ctx->pix_fmt;
    enc->frame->width = WINDOW_WIDTH;
    enc->frame->height = WINDOW_HEIGHT;
    r = av_frame_get_buffer(enc->frame, 0);
    if (r < 0) {
        SDL_LogError(SDL_LOG_CATEGORY_ERROR, "Could not allocate frame buffer\n%s\n", av_err2str(r));
        return -1;
    }

    enc->packet = av_packet_alloc();
    if (!enc->packet) {
        SDL_LogError(SDL_LOG_CATEGORY_ERROR, "Could not allocate packet buffer\n");
        return -1;
    }
    enc->pts = 0;

    return 0;
}

int video_encoder_send_frame(struct video_encoder* enc, const uint8_t* grayscale_data) {
    av_frame_make_writable(enc->frame);

    for (int y = 0; y < WINDOW_HEIGHT; y++) {
        SDL_memcpy(enc->frame->data[0] + y * enc->frame->linesize[0],
               grayscale_data + y * WINDOW_WIDTH, WINDOW_WIDTH);
    }
    for (int y = 0; y < WINDOW_HEIGHT / 2; y++) {
        SDL_memset(enc->frame->data[1] + y * enc->frame->linesize[1], 128, WINDOW_WIDTH / 2);
        SDL_memset(enc->frame->data[2] + y * enc->frame->linesize[2], 128, WINDOW_WIDTH / 2);
    }

    enc->frame->pts = enc->pts++;

    if (avcodec_send_frame(enc->codec_ctx, enc->frame) < 0) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "Error sending frame to encoder\n");
        return -1;
    }

    while (avcodec_receive_packet(enc->codec_ctx, enc->packet) == 0) {
        enc->packet->stream_index = enc->stream->index;
        av_packet_rescale_ts(enc->packet, enc->codec_ctx->time_base, enc->stream->time_base);
        av_interleaved_write_frame(enc->format_ctx, enc->packet);
        av_packet_unref(enc->packet);
    }
    return 0;
}

void video_encoder_finish(struct video_encoder* enc) {
    avcodec_send_frame(enc->codec_ctx, NULL);
    while (avcodec_receive_packet(enc->codec_ctx, enc->packet) == 0) {
        enc->packet->stream_index = enc->stream->index;
        av_packet_rescale_ts(enc->packet, enc->codec_ctx->time_base, enc->stream->time_base);
        av_interleaved_write_frame(enc->format_ctx, enc->packet);
        av_packet_unref(enc->packet);
    }

    av_write_trailer(enc->format_ctx);
    avcodec_free_context(&enc->codec_ctx);
    av_frame_free(&enc->frame);
    av_packet_free(&enc->packet);
    avio_close(enc->format_ctx->pb);
    avformat_free_context(enc->format_ctx);
}