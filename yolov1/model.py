import torch as nn


architecture_yolov1 = [
        # (kernel_size, filters, stride, padding)
        (7, 64, 2, 3),
        "M",
        (3, 192, 1, 1),
        "M",
        (1, 128, 1, 0),
        (3, 256, 1, 1),
        (1, 256, 1, 0),
        (3, 512, 1, 1),
        "M",
        # List of tuples: (kernel_size, filters, stride, padding, repeat)
        [(1, 256, 1, 0), (3, 512, 1, 1), 4],
        (1, 512, 1, 0),
        (3, 1024, 1, 1),
        "M",
        [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
        (3, 1024, 1, 1),
        (3, 1024, 2, 1),
        (3, 1024, 1, 1),
        (3, 1024, 1, 1),
        (3, 1024, 1, 1),
]


class CustomCNN(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CustomCNN, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))


class YOLOv1(nn.Module):
    def __init__(self, in_channels: int = 3, **kwargs):
        super(YOLOv1, self).__init__()
        self.architecture = architecture_yolov1
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(self.architecture)
        self.fcs = self._create_fcs(**kwargs)

    def forward(self, x):
        x = self.darknet(x)
        return self.fcs(torch.flatten(x, start_dim=1))

    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == tuple:
                layers += [
                    CustomCNN(
                        in_channels=in_channels,
                        out_channels=x[1],
                        kernel_size=x[0],
                        stride=x[2],
                        padding=x[3],
                    )
                ]
            elif type(x) == str:
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]
            elif type(x) == list:
                conv1: tuple = x[0]
                conv2: tuple = x[1]
                num_repeats: list = x[2]

                for _ in range(num_repeats):
                    layers += [
                        CustomCNN(
                            in_channels=in_channels,
                            out_channels=conv1[1],
                            kernel_size=conv1[0],
                            stride=conv1[2],
                            padding=conv1[3],
                        )
                    ]
                    layers += [
                        CustomCNN(
                            in_channels=conv1[1],
                            out_channels=conv2[1],
                            kernel_size=conv2[0],
                            stride=conv2[2],
                            padding=conv2[3],
                        )
                    ]
                    in_channels = conv2[1]

            # * will unpack the list
        return nn.Sequential(*layers)

    def _create_fcs(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 496),
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(496, S * S * (C + B * 5)),
        )
    
    def test(split_size, num_boxes, num_classes):
        model = YOLOv1(split_size, num_boxes, num_classes)
        x = torch.randn((2, 3, 448, 448))
        print(model(x).shape)
        assert model(x).shape == (2, 7 * 7 * 30)
        print("Success!")
