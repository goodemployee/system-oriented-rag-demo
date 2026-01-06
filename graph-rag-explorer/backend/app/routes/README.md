Routes 路徑介紹

它是：

    對外介面層（HTTP / API）

    將請求轉交至 application services

    負責請求與回應的格式轉換

它不是：

    不是 application（不包含流程邏輯）

    不是 core/domain（不處理核心規則）

    不是 infrastructure（不直接管理資源）