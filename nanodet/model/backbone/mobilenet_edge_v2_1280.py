import torch
import torch.nn as nn
import math

def conv_1x1_bn(inp, oup, stride, activation):
    total_padding = (1 - 1) // 2
  
    if activation == True:
      return nn.Sequential(
          nn.Conv2d(inp, oup, 1, stride, total_padding, bias=False),
          nn.BatchNorm2d(oup),
          nn.ReLU6()
      ) 
    else:  
      return nn.Sequential(
          nn.Conv2d(inp, oup, 1, stride, total_padding, bias=False),
          #nn.BatchNorm2d(oup),
      )


def conv_3x3_bn(inp, oup, stride, activation):
    total_padding = (3 - 1) // 2

    if activation == True:
      return nn.Sequential(
          nn.Conv2d(inp, oup, 3, stride, total_padding, bias=False),
          nn.BatchNorm2d(oup),
          nn.ReLU6()
      )
    else:
      return nn.Sequential(
          nn.Conv2d(inp, oup, 3, stride, total_padding, bias=False),
          #nn.BatchNorm2d(oup)
      )    

def conv_5x5_bn(inp, oup, stride, activation):
    total_padding = (5 - 1) // 2

    if activation == True:
      return nn.Sequential(
          nn.Conv2d(inp, oup, 5, stride, total_padding, bias=False),
          nn.BatchNorm2d(oup),
          nn.ReLU6()
      )
    else:    
      return nn.Sequential(
          nn.Conv2d(inp, oup, 5, stride, total_padding, bias=False),
          #nn.BatchNorm2d(oup),
      )

def depth_bn(inp, oup, stride, activation):
    total_padding = (3 - 1) // 2
    
    if activation == True:
      return nn.Sequential( 
          nn.Conv2d(inp, oup, 3, stride, total_padding, bias=False, groups = inp),
          nn.BatchNorm2d(oup),
          nn.ReLU6()
      ) 
    
    else:
      return nn.Sequential( 
          nn.Conv2d(inp, oup, 3, stride, total_padding, bias=False, groups = inp),
          #nn.BatchNorm2d(oup)
      )      
      
class MobileNetEdgeV21280(nn.Module):    
  def __init__(self, model_name, out_stages=(1, 3, 5), activation="ReLU6", pretrain=True):
    super(MobileNetEdgeV21280, self).__init__()
    print("Hello")

    self.stem = nn.Sequential(
        conv_5x5_bn(3, 64, 2, True),
        conv_1x1_bn(64, 24, 1, False),
        conv_3x3_bn(24, 192, 2, True),
        conv_1x1_bn(192, 48, 1, False)
    )

    self.b0_0 = conv_3x3_bn(16, 64, 1, True)
    self.b0_1 = conv_3x3_bn(16, 64, 1, True)
    self.b0_2 = conv_3x3_bn(16, 64, 1, True)
    self.b0_3 = conv_1x1_bn(192, 48, 1, False)

    self.b1_0 = conv_3x3_bn(48, 384, 1, True)
    self.b1_1 = conv_1x1_bn(384, 64, 1, False)

    self.b2_0 = conv_3x3_bn(16, 64, 1, True)
    self.b2_1 = conv_3x3_bn(16, 64, 1, True)
    self.b2_2 = conv_3x3_bn(16, 64, 1, True)
    self.b2_3 = conv_3x3_bn(16, 64, 1, True)
    self.b2_4 = conv_1x1_bn(256, 64, 1, False)

    self.b3_0 = conv_3x3_bn(64, 256, 1, True)
    self.b3_1 = conv_1x1_bn(256, 64, 1, False)

    self.b4_0 = conv_3x3_bn(16, 64, 1, True)
    self.b4_1 = conv_3x3_bn(16, 64, 1, True)
    self.b4_2 = conv_3x3_bn(16, 64, 1, True)
    self.b4_3 = conv_3x3_bn(16, 64, 1, True)
    self.b4_4 = conv_1x1_bn(256, 64, 1, False)

    self.b5_0 = conv_3x3_bn(64, 512, 1, True)
    self.b5_1 = conv_1x1_bn(512, 128, 1, False)

    self.b6_0 = conv_1x1_bn(128, 512, 1, True)
    self.b6_1 = depth_bn(512, 512, 1, True)
    self.b6_2 = conv_1x1_bn(512, 128, 1, False)

    self.b7_0 = conv_1x1_bn(128, 512, 1, True)
    self.b7_1 = depth_bn(512, 512, 1, True)
    self.b7_2 = conv_1x1_bn(512, 128, 1, False)

    self.b8_0 = conv_1x1_bn(128, 512, 1, True)
    self.b8_1 = depth_bn(512, 512, 1, True)
    self.b8_2 = conv_1x1_bn(512, 128, 1, False)

    self.b9_0 = conv_1x1_bn(128, 1024, 1, True)
    self.b9_1 = depth_bn(1024, 1024, 1, True)
    self.b9_2 = conv_1x1_bn(1024, 160, 1, False)

    self.b10_0 = conv_1x1_bn(160, 640, 1, True)
    self.b10_1 = depth_bn(640, 640, 1, True)
    self.b10_2 = conv_1x1_bn(640, 160, 1, False)

    self.b11_0 = conv_1x1_bn(160, 640, 1, True)
    self.b11_1 = depth_bn(640, 640, 1, True)
    self.b11_2 = conv_1x1_bn(640, 160, 1, False)

    self.b12_0 = conv_1x1_bn(160, 640, 1, True)
    self.b12_1 = depth_bn(640, 640, 1, True)
    self.b12_2 = conv_1x1_bn(640, 160, 1, False)    

    self.b13_0 = conv_1x1_bn(160, 1280, 1, True)
    self.b13_1 = depth_bn(1280, 1280, 1, True)
    self.b13_2 = conv_1x1_bn(1280, 192, 1, False)

    self.b14_0 = conv_1x1_bn(192, 768, 1, True)
    self.b14_1 = depth_bn(768, 768, 1, True)
    self.b14_2 = conv_1x1_bn(768, 192, 1, False)

    self.b15_0 = conv_1x1_bn(192, 768, 1, True)
    self.b15_1 = depth_bn(768, 768, 1, True)
    self.b15_2 = conv_1x1_bn(768, 192, 1, False)

    self.b16_0 = conv_1x1_bn(192, 768, 1, True)
    self.b16_1 = depth_bn(768, 768, 1, True)
    self.b16_2 = conv_1x1_bn(768, 192, 1, False)    

    self.b17_0 = conv_1x1_bn(192, 1536, 1, True)
    self.b17_1 = depth_bn(1536, 1536, 1, True)
    self.b17_2 = conv_1x1_bn(1536, 256, 1, False)   
    self.b17_3 = conv_1x1_bn(256, 1280, 1, False)

    self._initialize_weights()


  def forward(self, x): 
      output = []
      x = self.stem(x)
            
      #Block
      (s1,s2,s3) = torch.split(x, 16, 1)

      s1 = self.b0_0(s1)
      s2 = self.b0_1(s2)
      s3 = self.b0_2(s3)

      b = torch.cat((s1,s2,s3), 1)
      b = self.b0_3(b)
      x = torch.add(x,b)

      #Block 1
      x = self.b1_0(x)
      x = self.b1_1(x)
      
      
      #Block 2        
      (s1,s2,s3,s4) = torch.split(x, 16, 1)
      
      s1 = self.b2_0(s1)
      s2 = self.b2_1(s2)
      s3 = self.b2_2(s3)
      s4 = self.b2_3(s4)

      b = torch.concat((s1,s2,s3,s4), 1)
  
      b = self.b2_4(b)

      print(b.size())
      
      x = torch.add(x,b)

      
      #Block 3          
      b = self.b3_0(x)
      b = self.b3_1(b)
      x = torch.add(x,b)


      #Block 4  
      (s1,s2,s3,s4) = torch.split(x, 16, 1)
      
      s1 = self.b4_0(s1)
      s2 = self.b4_1(s2)
      s3 = self.b4_2(s3)
      s4 = self.b4_3(s4)

      b = torch.concat((s1,s2,s3,s4), 1)
      
      b = self.b4_4(b)
      
      x = torch.add(x,b)

      print("Level 1 " + str(x.size()))
      output.append(x)

      #Block 5
      x = self.b5_0(x)
      x = self.b5_1(x)  

      #Block 6
      b = self.b6_0(x)
      b = self.b6_1(b)
      b = self.b6_2(b)
      x = torch.add(x,b)

      #Block 7
      b = self.b7_0(x)
      b = self.b7_1(b)
      b = self.b7_2(b)
      x = torch.add(x,b)

      #Block 8
      b = self.b8_0(x)
      b = self.b8_1(b)
      b = self.b8_2(b)
      x = torch.add(x,b)
      
      #Block 9
      x = self.b9_0(x)
      x = self.b9_1(x)
      x = self.b9_2(x)

      #Block 10
      b = self.b10_0(x)
      b = self.b10_1(b)
      b = self.b10_2(b)
      x = torch.add(x,b)

      #Block 11
      b = self.b11_0(x)
      b = self.b11_1(b)
      b = self.b11_2(b)
      x = torch.add(x,b)

      #Block 12
      b = self.b12_0(x)
      b = self.b12_1(b)
      b = self.b12_2(b)
      x = torch.add(x,b) 

      print("Level 2 " + str(x.size()))
      output.append(x)

      #Block 13
      x = self.b13_0(x)
      x = self.b13_1(x)
      x = self.b13_2(x)
      
      #Block 14
      b = self.b14_0(x)
      b = self.b14_1(b)
      b = self.b14_2(b)
      x = torch.add(x, b)

      #Block 15
      b = self.b15_0(x)
      b = self.b15_1(b)
      b = self.b15_2(b)
      x = torch.add(x, b)

      #Block 16
      b = self.b16_0(x)
      b = self.b16_1(b)
      b = self.b16_2(b)
      x = torch.add(x, b)

      #Block 17
      x = self.b17_0(x)
      x = self.b17_1(x)
      x = self.b17_2(x)
      #x = self.b17_3(x)

      print("Level 3 " + str(x.size()))
      output.append(x)

      return output #x

  def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.001)
                m.bias.data.zero_()
    
  def load_pretrain(self, path):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict, strict=True)
