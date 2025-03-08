// Copyright (c) 2025 Minmin Gong
//

uint32_t main(uint32_t face_id : SV_PrimitiveID) : SV_Target0
{
    return face_id + 1; // 0 is reserved as empty
}
